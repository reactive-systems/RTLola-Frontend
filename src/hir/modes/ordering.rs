use crate::common_ir::{Layer, SRef, StreamLayers};

use super::{EdgeWeight, EvaluationOrder};

use super::dg_functionality::*;
use std::collections::HashMap;

use crate::hir::modes::{dependencies::DependenciesAnalyzed, ir_expr::WithIrExpr, types::TypeChecked, HirMode};
use crate::hir::Hir;
use petgraph::{algo::is_cyclic_directed, Graph, Outgoing};

pub(crate) trait EvaluationOrderBuilt {
    fn layers(&self, sr: SRef) -> StreamLayers;
}

impl EvaluationOrderBuilt for EvaluationOrder {
    fn layers(&self, sr: SRef) -> StreamLayers {
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

impl<A: OrderedWrapper<InnerO = T>, T: EvaluationOrderBuilt + 'static> EvaluationOrderBuilt for A {
    fn layers(&self, sr: SRef) -> StreamLayers {
        self.inner_order().layers(sr)
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
        M: WithIrExpr + HirMode + 'static + DependenciesAnalyzed + TypeChecked,
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
        M: WithIrExpr + HirMode + 'static + DependenciesAnalyzed + TypeChecked,
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
    use crate::common_ir::{Layer, SRef};
    use crate::hir::StreamLayers;
    use std::collections::HashMap;
    fn check_eval_order_for_spec(
        _spec: &str,
        _event_layers: HashMap<SRef, StreamLayers>,
        _periodic_layers: HashMap<SRef, StreamLayers>,
    ) {
        todo!()
    }

    #[test]
    #[ignore]
    fn basic_spec() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let name_mapping =
            vec![("a", SRef::InRef(1)), ("b", SRef::OutRef(1))].into_iter().collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (name_mapping["a"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (name_mapping["b"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = HashMap::new();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }

    #[test]
    #[ignore]
    fn simple_spec() {
        let spec =
            "input a : Int8 input b :Int8\noutput c @2Hz := a.hold().defaults(to: 0) + 3\noutput d @1Hz := a.hold().defaults(to: 0) + c\noutput e := a + b";
        let name_mapping = vec![
            ("a", SRef::InRef(1)),
            ("b", SRef::InRef(1)),
            ("c", SRef::OutRef(1)),
            ("d", SRef::OutRef(2)),
            ("e", SRef::OutRef(3)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let event_layers = vec![
            (name_mapping["a"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (name_mapping["b"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (name_mapping["e"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        let periodic_layers = vec![
            (name_mapping["c"], StreamLayers::new(Layer::new(0), Layer::new(1))),
            (name_mapping["d"], StreamLayers::new(Layer::new(0), Layer::new(2))),
        ]
        .into_iter()
        .collect();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
}
