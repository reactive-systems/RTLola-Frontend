use super::EdgeWeight;
use super::SRef;
use crate::hir::modes::{dependencies::DependenciesAnalyzed, ir_expr::WithIrExpr, types::TypeChecked};
use crate::hir::HirMode;
use crate::Hir;
use petgraph::Graph;

pub(crate) fn graph_without_negative_offset_edges(graph: &Graph<SRef, EdgeWeight>) -> Graph<SRef, EdgeWeight> {
    let mut working_graph = graph.clone();
    graph.edge_indices().for_each(|edge_index| match graph.edge_weight(edge_index).unwrap() {
        EdgeWeight::Offset(o) if *o < 0 => {
            working_graph.remove_edge(edge_index);
        }
        _ => {}
    });
    working_graph
}

pub(crate) fn graph_without_close_edges(graph: &Graph<SRef, EdgeWeight>) -> Graph<SRef, EdgeWeight> {
    let mut working_graph = graph.clone();
    graph.edge_indices().for_each(|edge_index| {
        let edge_weight = graph.edge_weight(edge_index).unwrap();
        let (edge_src, edge_tar) = graph.edge_endpoints(edge_index).unwrap();
        match (edge_weight, edge_src == edge_tar) {
            (EdgeWeight::Close(_), true) => {
                working_graph.remove_edge(edge_index);
            }
            _ => {}
        }
    });
    working_graph
}

pub(crate) fn only_spawn_edges(graph: &Graph<SRef, EdgeWeight>) -> Graph<SRef, EdgeWeight> {
    let mut working_graph = graph.clone();
    working_graph.edge_indices().for_each(|edge_index| {
        let test = graph.edge_weight(edge_index).unwrap();
        if let EdgeWeight::Spawn(_) = test {
        } else {
            working_graph.remove_edge(edge_index);
        }
    });
    working_graph
}

pub(crate) fn split_graph<M>(
    spec: &Hir<M>,
    graph: Graph<SRef, EdgeWeight>,
) -> (Graph<SRef, EdgeWeight>, Graph<SRef, EdgeWeight>)
where
    M: WithIrExpr + HirMode + 'static + DependenciesAnalyzed + TypeChecked,
{
    // remove edges and nodes, so mapping does not change
    let mut event_graph = graph.clone();
    let mut periodic_graph = graph.clone();
    // delete edges -> TODO I am not sure if this is a good idea to delete edges while iterating over them
    graph.edge_indices().for_each(|edge_index| {
        let (src, tar) = graph.edge_endpoints(edge_index).unwrap();
        let src = graph.node_weight(src).unwrap();
        let tar = graph.node_weight(tar).unwrap();
        match (spec.is_event(*src), spec.is_event(*tar)) {
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
        let node_sref = graph.node_weight(node_index).unwrap();
        if spec.is_event(*node_sref) {
            periodic_graph.remove_node(node_index);
        } else {
            assert!(spec.is_periodic(*node_sref));
            event_graph.remove_node(node_index);
        }
    });
    (event_graph, periodic_graph)
}
