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
        let _event_layers = Self::compute_layers(spec, &event_graph)?;
        let _periodic_layers = Self::compute_layers(spec, &periodic_graph)?;
        // Ok(EvaluationOrder { event_layers, periodic_layers })
        todo!()
    }

    fn compute_layers<M>(
        spec: &Hir<M>,
        graph: &Graph<SRef, EdgeWeight>,
    ) -> Result<HashMap<SRef, Layer>>
    where
        M: WithIrExpr + HirMode + 'static + DependenciesAnalyzed + TypeChecked,
    {
        if is_cyclic_directed(&graph) {
            return Err(OrderingErr::Cycle);
        }
        let mut sref_to_layer = spec.inputs().map(|i| (i.sr, Layer::new(0))).collect::<HashMap<SRef, Layer>>();
        while graph.node_count() != sref_to_layer.len() {
            graph.node_indices().for_each(|node| {
                let sref = graph.node_weight(node).unwrap();
                if !sref_to_layer.contains_key(sref) {
                    //Layer for current streamcheck incoming
                    let layer = graph
                        .neighbors_directed(node, Outgoing)//or incoming -> try
                        .map(|outgoing_neighbor| sref_to_layer.get(&graph.node_weight(outgoing_neighbor).unwrap()).map(|layer| *layer))
                        .fold(Some(Layer::new(0)), |cur_res, neighbor_layer| match (cur_res, neighbor_layer) {
                            (Some(layer1), Some(layer2)) => Some(std::cmp::max(layer1, layer2)),
                            _ => None,
                        });
                    if let Some(layer) = layer {
                        sref_to_layer.insert(*sref, layer);
                    }
                }
            });
        }
        Ok(sref_to_layer)
    }
}

#[cfg(test)]
mod tests {
    use crate::common_ir::{Layer, SRef};
    use std::collections::HashMap;
    fn check_eval_order_for_spec(
        _spec: &str,
        _event_layers: HashMap<SRef, Layer>,
        _periodic_layers: HashMap<SRef, Layer>,
    ) {
        todo!()
    }

    #[test]
    #[ignore]
    fn simple_spec() {
        let spec = "input a: UInt8\noutput b: UInt8 := a";
        let event_layers = todo!();
        let periodic_layers = todo!();
        check_eval_order_for_spec(spec, event_layers, periodic_layers)
    }
}
