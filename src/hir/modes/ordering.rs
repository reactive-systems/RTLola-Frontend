use crate::common_ir::{Layer, SRef, StreamLayers};

use super::{EdgeWeight, EvaluationOrder};

use super::dg_functionality::*;
use std::collections::HashMap;

use crate::hir::modes::{dependencies::WithDependencies, ir_expr::WithIrExpr, types::TypeChecked, HirMode, Ordered};
use crate::hir::Hir;
use petgraph::{algo::is_cyclic_directed, Graph, Outgoing};

pub(crate) trait EvaluationOrderBuilt {
    fn stream_layers(&self, sr: SRef) -> StreamLayers;
}

impl EvaluationOrderBuilt for EvaluationOrder {
    fn stream_layers(&self, sr: SRef) -> StreamLayers {
        match self.event_layers.get(&sr) {
            Some(layer) => *layer,
            None => self.periodic_layers[&sr],
        }
        // todo!("Is there a better way to decide if the stream is periodic or event-based?")
    }
}

pub(crate) trait OrderedWrapper {
    type InnerO: EvaluationOrderBuilt;
    fn inner_order(&self) -> &Self::InnerO;
}

impl OrderedWrapper for Ordered {
    type InnerO = EvaluationOrder;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}

impl<A: OrderedWrapper<InnerO = T>, T: EvaluationOrderBuilt + 'static> EvaluationOrderBuilt for A {
    fn stream_layers(&self, sr: SRef) -> StreamLayers {
        self.inner_order().stream_layers(sr)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OrderingErr {
    Cycle,
}

pub(crate) struct OrderingReport {}

type Result<T> = std::result::Result<T, OrderingErr>;

impl EvaluationOrder {
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Result<EvaluationOrder>
    where
        M: WithIrExpr + HirMode + 'static + WithDependencies + TypeChecked,
    {
        // Compute Evaluation Layers
        let graph = graph_without_negative_offset_edges(spec.graph());
        let graph = graph_without_close_edges(&graph);
        // split graph in periodic and event-based
        let (event_graph, periodic_graph) = split_graph(spec, graph);
        let event_layers = Self::compute_layers(spec, &event_graph)?;
        let periodic_layers = Self::compute_layers(spec, &periodic_graph)?;
        Ok(EvaluationOrder { event_layers, periodic_layers })
    }

    fn compute_layers<M>(spec: &Hir<M>, graph: &Graph<SRef, EdgeWeight>) -> Result<HashMap<SRef, StreamLayers>>
    where
        M: WithIrExpr + HirMode + 'static + WithDependencies + TypeChecked,
    {
        debug_assert!(is_cyclic_directed(&graph), "This should be already checked in the dependency analysis.");
        let graph_with_only_spawn_edges = only_spawn_edges(graph);
        let mut evaluation_layers = spec.inputs().map(|i| (i.sr, Layer::new(0))).collect::<HashMap<SRef, Layer>>();
        let mut spawn_layers = HashMap::<SRef, Layer>::new();
        while graph.node_count() != evaluation_layers.len() {
            // build spawn layers
            graph_with_only_spawn_edges.node_indices().for_each(|node| {
                let sref = graph.node_weight(node).unwrap();
                if !spawn_layers.contains_key(sref) {
                    let computed_spawn_layer = graph_with_only_spawn_edges
                        .neighbors_directed(node, Outgoing)
                        .flat_map(|outgoing_neighbor| {
                            evaluation_layers
                                .get(&graph_with_only_spawn_edges.node_weight(outgoing_neighbor).unwrap())
                                .copied()
                        })
                        .fold(Some(Layer::new(0)), |cur_res, neighbor_layer| {
                            if let Some(cur_res_layer) = cur_res {
                                Some(std::cmp::max(cur_res_layer, neighbor_layer))
                            } else {
                                None
                            }
                        });
                    if let Some(layer) = computed_spawn_layer {
                        spawn_layers.insert(*sref, layer);
                    }
                }
            });
            graph.node_indices().for_each(|node| {
                let sref = graph.node_weight(node).unwrap();
                // build evaluation layers
                if !evaluation_layers.contains_key(sref) && spawn_layers.contains_key(sref){
                    //Layer for current streamcheck incoming
                    let computed_spawn_layer = graph
                        .neighbors_directed(node, Outgoing)//or incoming -> try
                        .flat_map(|outgoing_neighbor| if outgoing_neighbor == node {None} else {Some(outgoing_neighbor)}) //delete self references
                        .flat_map(|outgoing_neighbor| evaluation_layers.get(&graph.node_weight(outgoing_neighbor).unwrap()).copied())
                        .fold(Some(Layer::new(0)), |cur_res, neighbor_layer| {
                        if let Some(cur_res_layer) = cur_res { Some(std::cmp::max(cur_res_layer, neighbor_layer))} else {None}
                    });
                        if let Some(layer) = computed_spawn_layer {
                            if spawn_layers[sref] < layer {evaluation_layers.insert(*sref, layer);}
                    }
                }
            });
        }
        Ok(evaluation_layers
            .into_iter()
            .map(|(key, evaluation_layer)| (key, StreamLayers::new(spawn_layers[&key], evaluation_layer)))
            .collect::<HashMap<SRef, StreamLayers>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::modes::IrExpression;
    use crate::parse::parse;
    use crate::reporting::Handler;
    use crate::FrontendConfig;
    use std::path::PathBuf;
    #[allow(dead_code, unreachable_code, unused_variables)]
    fn check_eval_order_for_spec(
        spec: &str,
        ref_event_layers: HashMap<SRef, StreamLayers>,
        ref_periodic_layers: HashMap<SRef, StreamLayers>,
    ) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let _hir = Hir::<IrExpression>::transform_expressions(ast, &handler, &config);
        let order: EvaluationOrder = todo!();
        let EvaluationOrder { event_layers, periodic_layers } = order;
        assert_eq!(event_layers.len(), ref_event_layers.len());
        event_layers.iter().for_each(|(sr, layers)| {
            let ref_layers = &ref_event_layers[sr];
            assert_eq!(ref_layers.spawn_layer(), layers.spawn_layer());
            assert_eq!(ref_layers.evaluation_layer(), layers.evaluation_layer());
        });
        assert_eq!(periodic_layers.len(), ref_periodic_layers.len());
        periodic_layers.iter().for_each(|(sr, layers)| {
            let ref_layers = &ref_periodic_layers[sr];
            assert_eq!(ref_layers.spawn_layer(), layers.spawn_layer());
            assert_eq!(ref_layers.evaluation_layer(), layers.evaluation_layer());
        });
    }

    #[test]
    #[ignore]
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn hold_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.hold().defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers =
            vec![(sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0)))].into_iter().collect();
        let periodic_layers =
            vec![(sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1)))].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.aggregate(over: 5, using: sum)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn offset_lookups() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by:-1).defaults(to: 0)\noutput c: UInt8 := a.offset(by:-2).defaults(to: 0)\noutput d: UInt8 := a.offset(by:-3).defaults(to: 0)\noutput e: UInt8 := a.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::OutRef(0)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
    #[test]
    #[ignore]
    fn negative_loop_different_offsets() {
        let spec = "input a: Int8\noutput b: Int8 := a.offset(by: -1).defaults(to: 0) + d.offset(by:-2).defaults(to:0)\noutput c: Int8 := b.offset(by:-3).defaults(to: 0)\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn lookup_chain() {
        let spec = "input a: Int8\noutput b: Int8 := a + d.hold().defaults(to:0)\noutput c: Int8 := b\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    fn multiple_input_stream() {
        let spec = "input a: Int8\ninput b: Int8\noutput c: Int8 := a + b.hold().defaults(to:0)\noutput d: Int8 := a + c.offset(by: -1).defaults(to: 0)\noutput e: Int8 = c + 3\noutput f: Int8 = c + 6\noutput g: Int8 := b + 3\noutput h: Int8 = g + f";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
            ("f", SRef::OutRef(3)),
            ("g", SRef::OutRef(4)),
            ("h", SRef::OutRef(5)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["g"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["g"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn event_and_periodic_stream_mix() {
        let spec =
            "input a : Int8 \ninput b :Int8\noutput c @2Hz := a.hold().defaults(to: 0) + 3\noutput d @1Hz := a.hold().defaults(to: 0) + c\noutput e := a + b";
        let sname_to_sref = vec![
            ("a", SRef::InRef(1)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
}
