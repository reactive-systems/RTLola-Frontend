use super::{DependencyGraph, EdgeWeight};
use crate::hir::modes::{dependencies::DepAnaTrait, ir_expr::IrExprTrait, types::TypedTrait};
use crate::hir::HirMode;
use crate::Hir;

pub(crate) fn graph_without_negative_offset_edges(graph: &DependencyGraph) -> DependencyGraph {
    let mut working_graph = graph.clone();
    graph.edge_indices().for_each(|edge_index| {
        if has_negative_offset(graph.edge_weight(edge_index).unwrap()) {
            working_graph.remove_edge(edge_index);
        }
    });
    working_graph
}

fn has_negative_offset(e: &EdgeWeight) -> bool {
    match e {
        EdgeWeight::Offset(o) if *o < 0 => true,
        EdgeWeight::Spawn(s) => has_negative_offset(s),
        EdgeWeight::Filter(f) => has_negative_offset(f),
        EdgeWeight::Close(c) => has_negative_offset(c),
        _ => false,
    }
}

pub(crate) fn graph_without_close_edges(graph: &DependencyGraph) -> DependencyGraph {
    let mut working_graph = graph.clone();
    graph.edge_indices().for_each(|edge_index| {
        if let EdgeWeight::Close(_) = graph.edge_weight(edge_index).unwrap() {
            working_graph.remove_edge(edge_index);
        }
    });
    working_graph
}

pub(crate) fn only_spawn_edges(graph: &DependencyGraph) -> DependencyGraph {
    let mut working_graph = graph.clone();
    graph.edge_indices().for_each(|edge_index| {
        if let EdgeWeight::Spawn(_) = graph.edge_weight(edge_index).unwrap() {
        } else {
            let res = working_graph.remove_edge(edge_index);
            assert!(res.is_some());
        }
    });
    working_graph
}

pub(crate) fn split_graph<M>(spec: &Hir<M>, graph: DependencyGraph) -> (DependencyGraph, DependencyGraph)
where
    M: IrExprTrait + HirMode + 'static + DepAnaTrait + TypedTrait,
{
    // remove edges and nodes, so mapping does not change
    let mut event_graph = graph.clone();
    let mut periodic_graph = graph.clone();
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
