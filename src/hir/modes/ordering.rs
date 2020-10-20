use crate::common_ir::{Layer, SRef};

use super::{EdgeWeight, EvaluationOrder};

use super::common_functionality::graph_without_negative_offset_edges;
use std::collections::HashMap;

use crate::hir::modes::{dependencies::DependenciesAnalyzed, ir_expr::WithIrExpr, types::TypeChecked, HirMode};
use crate::hir::Hir;
use petgraph::{algo::is_cyclic_directed, graph::EdgeIndex, graph::NodeIndex, Graph, Outgoing};

pub(crate) trait EvaluationOrderBuilt {
    fn layer(&self, sr: SRef) -> Layer;
}

impl EvaluationOrderBuilt for EvaluationOrder {
    fn layer(&self, _sr: SRef) -> Layer {
        todo!()
    }
}

pub(crate) trait OrderedWrapper {
    type InnerO: EvaluationOrderBuilt;
    fn inner_order(&self) -> &Self::InnerO;
}

impl<A: OrderedWrapper<InnerO = T>, T: EvaluationOrderBuilt + 'static> EvaluationOrderBuilt for A {
    fn layer(&self, sr: SRef) -> Layer {
        self.inner_order().layer(sr)
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
        let node_mapping = spec
            .graph()
            .node_indices()
            .map(|node| {
                let weight = spec.graph().node_weight(node).unwrap();
                (node, *weight)
            })
            .collect::<HashMap<NodeIndex, SRef>>();
        let edge_mapping_edge_to_index = spec
            .graph()
            .edge_indices()
            .map(|edge| {
                let (src, tar) = spec.graph().edge_endpoints(edge).unwrap();
                let weight = spec.graph().edge_weight(edge).unwrap();
                ((node_mapping[&src], *weight, node_mapping[&tar]), edge)
            })
            .collect::<HashMap<(SRef, EdgeWeight, SRef), EdgeIndex>>();
        let edge_mapping_index_to_edge =
            edge_mapping_edge_to_index.iter().map(|(edge, index)| (*index, *edge)).collect();
        let graph = graph_without_negative_offset_edges(spec.graph(), &edge_mapping_edge_to_index);
        // split graph in periodic and event-based
        let (event_graph, periodic_graph) = Self::split_graph(spec, graph, &node_mapping, &edge_mapping_index_to_edge);
        let event_layers = Self::compute_layer(spec, &event_graph, &node_mapping)?;
        let periodic_layers = Self::compute_layer(spec, &periodic_graph, &node_mapping)?;
        Ok(EvaluationOrder { event_layers, periodic_layers })
    }

    fn compute_layer<M>(
        spec: &Hir<M>,
        graph: &Graph<SRef, EdgeWeight>,
        node_mapping: &HashMap<NodeIndex, SRef>,
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
                let sref = &node_mapping[&node];
                if !sref_to_layer.contains_key(sref) {
                    //Layer for current streamcheck incoming
                    let layer = graph
                        .neighbors_directed(node, Outgoing)//or incoming -> try
                        .map(|outgoing_neighbor| sref_to_layer.get(&node_mapping[&outgoing_neighbor]).map(|layer| *layer))
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

    fn split_graph<M>(
        spec: &Hir<M>,
        graph: Graph<SRef, EdgeWeight>,
        node_mapping: &HashMap<NodeIndex, SRef>,
        edge_mapping: &HashMap<EdgeIndex, (SRef, EdgeWeight, SRef)>,
    ) -> (Graph<SRef, EdgeWeight>, Graph<SRef, EdgeWeight>)
    where
        M: WithIrExpr + HirMode + 'static + DependenciesAnalyzed + TypeChecked,
    {
        // remove edges and nodes, so mapping does not change
        let mut event_graph = graph.clone();
        let mut periodic_graph = graph.clone();
        // delete edges
        graph.edge_indices().for_each(|edge_index| {
            let (src, _weight, tar) = edge_mapping[&edge_index];
            match (spec.is_event(src), spec.is_event(tar)) {
                (true, true) => {
                    periodic_graph.remove_edge(edge_index);
                }
                (false, false) => {
                    event_graph.remove_edge(edge_index);
                }
                _ => {
                    event_graph.remove_edge(edge_index);
                    periodic_graph.remove_edge(edge_index);
                }
            }
        });
        // delete nodes
        graph.node_indices().for_each(|node_index| {
            let node_sref = node_mapping[&node_index];
            if spec.is_event(node_sref) {
                periodic_graph.remove_node(node_index);
            } else {
                assert!(spec.is_periodic(node_sref));
                event_graph.remove_node(node_index);
            }
        });
        (event_graph, periodic_graph)
    }
}
