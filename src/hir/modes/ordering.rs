use crate::common_ir::{Layer, SRef, StreamLayers};

use super::EvaluationOrder;

use super::dg_functionality::*;
use std::collections::HashMap;

use crate::hir::modes::{
    dependencies::WithDependencies, ir_expr::WithIrExpr, types::TypeChecked, DependencyGraph, HirMode, Ordered,
};
use crate::hir::Hir;
use petgraph::{algo::is_cyclic_directed, Outgoing};

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

pub(crate) struct OrderingReport {}

impl EvaluationOrder {
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> EvaluationOrder
    where
        M: WithIrExpr + HirMode + 'static + WithDependencies + TypeChecked,
    {
        // Compute Evaluation Layers
        let graph = graph_without_negative_offset_edges(spec.graph());
        let graph = graph_without_close_edges(&graph);
        // split graph in periodic and event-based
        let (event_graph, periodic_graph) = split_graph(spec, graph);
        let event_layers = Self::compute_layers(spec, &event_graph, true);
        let periodic_layers = Self::compute_layers(spec, &periodic_graph, false);
        EvaluationOrder { event_layers, periodic_layers }
    }

    fn compute_layers<M>(spec: &Hir<M>, graph: &DependencyGraph, is_event: bool) -> HashMap<SRef, StreamLayers>
    where
        M: WithIrExpr + HirMode + 'static + WithDependencies + TypeChecked,
    {
        debug_assert!(!is_cyclic_directed(&graph), "This should be already checked in the dependency analysis.");
        let spawn_graph = only_spawn_edges(&graph_without_negative_offset_edges(graph));
        let mut evaluation_layers = if is_event {
            spec.inputs().map(|i| (i.sr, Layer::new(0))).collect::<HashMap<SRef, Layer>>()
        } else {
            HashMap::new()
        };
        let mut spawn_layers = if is_event {
            spec.inputs().map(|i| (i.sr, Layer::new(0))).collect::<HashMap<SRef, Layer>>()
        } else {
            HashMap::new()
        };
        while graph.node_count() != evaluation_layers.len() {
            // build spawn layers
            spawn_graph.node_indices().for_each(|node| {
                let sref = spawn_graph.node_weight(node).unwrap();
                if !spawn_layers.contains_key(sref) {
                    let neighbor_layers: Vec<_> = spawn_graph
                        .neighbors_directed(node, Outgoing)
                        .map(|outgoing_neighbor| {
                            evaluation_layers.get(&spawn_graph.node_weight(outgoing_neighbor).unwrap()).copied()
                        })
                        .collect();
                    let computed_spawn_layer = if neighbor_layers.is_empty() {
                        Some(Layer::new(0))
                    } else {
                        neighbor_layers
                            .into_iter()
                            .fold(Some(Layer::new(0)), |cur_res_layer, neighbor_layer| {
                                match (cur_res_layer, neighbor_layer) {
                                    (Some(cur_res_layer), Some(neighbor_layer)) => {
                                        Some(std::cmp::max(cur_res_layer, neighbor_layer))
                                    }
                                    _ => None,
                                }
                            })
                            .map(|layer| Layer::new(layer.inner() + 1))
                    };
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
                    let neighbor_layers : Vec<_> = graph
                    .neighbors_directed(node, Outgoing)//or incoming -> try
                    .flat_map(|outgoing_neighbor| if outgoing_neighbor == node {None} else {Some(outgoing_neighbor)}) //delete self references
                    .map(|outgoing_neighbor| evaluation_layers.get(&graph.node_weight(outgoing_neighbor).unwrap()).copied())
                    .collect();
                    let computed_evaluation_layer = if neighbor_layers.is_empty() {
                        if is_event {Some(Layer::new(1))} else {Some(Layer::new(0))}
                    } else {
                        neighbor_layers.into_iter().fold(Some(Layer::new(0)), |cur_res_layer, neighbor_layer| {
                            match (cur_res_layer, neighbor_layer) {
                                (Some(cur_res_layer), Some(neighbor_layer)) => Some(std::cmp::max(cur_res_layer, neighbor_layer)),
                                _ => None
                            }
                    }).map(|layer| Layer::new(layer.inner() +  1))
                    };
                    if let Some(layer) = computed_evaluation_layer {
                        let layer = if spawn_layers[sref] < layer {layer} else {Layer::new(spawn_layers[sref].inner() + 1)};
                        evaluation_layers.insert(*sref, layer);
                    }
                }
            });
        }
        evaluation_layers
            .into_iter()
            .map(|(key, evaluation_layer)| (key, StreamLayers::new(spawn_layers[&key], evaluation_layer)))
            .collect::<HashMap<SRef, StreamLayers>>()
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
    fn check_eval_order_for_spec(
        spec: &str,
        ref_event_layers: HashMap<SRef, StreamLayers>,
        ref_periodic_layers: HashMap<SRef, StreamLayers>,
    ) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<IrExpression>::from_ast(ast, &handler, &config)
            .build_dependency_graph()
            .unwrap()
            .type_check(&handler)
            .unwrap();
        let order = EvaluationOrder::analyze(&hir);
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
    fn synchronous_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a\noutput c:UInt8 := b";
        let sname_to_sref = vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn hold_lookup() {
        let spec =
            "input a: UInt8\ninput b:UInt8\noutput c: UInt8 := a.hold().defaults(to: 0) + b\noutput d: UInt8 := c.hold().defaults(to: 0) + a";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::InRef(1)), ("c", SRef::OutRef(0)), ("d", SRef::OutRef(1))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn offset_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 := a.offset(by: -1).defaults(to: 0)\noutput c: UInt8 := b.offset(by: -1).defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn sliding_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over: 1s, using: sum)\noutput c: UInt8 := a + 3\noutput d: UInt8 @1Hz := c.aggregate(over: 1s, using: sum)\noutput e: UInt8 := b + d\noutput f: UInt8 @2Hz := e.hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::OutRef(0)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
            ("f", SRef::OutRef(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn discrete_window_lookup() {
        let spec = "input a: UInt8\noutput b: UInt8 @1Hz := a.aggregate(over_discrete: 5, using: sum)\noutput c: UInt8 := a + 3\noutput d: UInt8 @1Hz := c.aggregate(over_discrete: 5, using: sum)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(1))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
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

    #[test]
    fn multiple_input_stream() {
        let spec = "input a: Int8\ninput b: Int8\noutput c: Int8 := a + b.hold().defaults(to:0)\noutput d: Int8 := a + c.offset(by: -1).defaults(to: 0)\noutput e: Int8 := c + 3\noutput f: Int8 := c + 6\noutput g: Int8 := b + 3\noutput h: Int8 := g + f";
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
            (sname_to_sref["h"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn event_and_periodic_stream_mix() {
        let spec =
            "input a : Int8 \ninput b :Int8\noutput c @2Hz := a.hold().defaults(to: 0) + 3\noutput d @1Hz := a.hold().defaults(to: 0) + c\noutput e := a + b";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
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
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn negative_and_postive_lookups_as_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a + d.offset(by:-1).defaults(to:0)\noutput c: Int8 := b\noutput d: Int8 := c";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
    #[test]
    fn sliding_windows_chain_and_hold_lookup() {
        let spec = "input a: Int8\noutput b@1Hz := a.aggregate(over: 1s, using: sum) + d.offset(by: -1).defaults(to: 0)\noutput c@2Hz := b.aggregate(over: 1s, using: sum)\noutput d@2Hz := b.hold().defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers =
            vec![(sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0)))].into_iter().collect();
        let periodic_layers = vec![
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(0), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn simple_chain_with_parameter() {
        let spec = "input a: Int8\noutput b := a + 5\noutput c(para) spawn with b := para + a";
        let sname_to_sref = vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(2), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn lookup_chain_with_parametrization() {
        let spec = "input a: Int8\noutput b(para) spawn with a if a > 6 := a + para\noutput c(para) spawn with a if a > 6 := a + b(a)\noutput d(para) spawn with a if a > 6 := a + c(a)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::OutRef(0)), ("c", SRef::OutRef(1)), ("d", SRef::OutRef(2))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(1), Layer::new(4))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_loop_with_lookup_in_close() {
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a if a < b := p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) < 4 := b + 5\noutput e(p) spawn with b := d(p).hold().defaults(to: 0) + b\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5\noutput g(p) spawn with b close f(p).hold().defaults(to: 0) < 6 := b + 5";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
            ("f", SRef::OutRef(3)),
            ("g", SRef::OutRef(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(3))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(4), Layer::new(5))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(1), Layer::new(6))),
            (sname_to_sref["f"], StreamLayers::new(Layer::new(1), Layer::new(7))),
            (sname_to_sref["g"], StreamLayers::new(Layer::new(1), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_nested_lookup_implicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
        let sname_to_sref =
            vec![("a", SRef::InRef(0)), ("b", SRef::InRef(1)), ("c", SRef::OutRef(0)), ("d", SRef::OutRef(1))]
                .into_iter()
                .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    fn parameter_nested_lookup_explicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::InRef(0)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(0)),
            ("d", SRef::OutRef(1)),
            ("e", SRef::OutRef(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (sname_to_sref["a"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["b"], StreamLayers::new(Layer::new(0), Layer::new(0))),
            (sname_to_sref["c"], StreamLayers::new(Layer::new(1), Layer::new(2))),
            (sname_to_sref["d"], StreamLayers::new(Layer::new(0), Layer::new(3))),
            (sname_to_sref["e"], StreamLayers::new(Layer::new(0), Layer::new(4))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![].into_iter().collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
}
