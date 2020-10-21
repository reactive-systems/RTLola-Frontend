use super::EdgeWeight;
use super::SRef;
use petgraph::{graph::EdgeIndex, Graph};
use std::collections::HashMap;

pub(crate) fn graph_without_negative_offset_edges(
    graph: &Graph<SRef, EdgeWeight>,
    edge_mapping: &HashMap<(SRef, &EdgeWeight, SRef), EdgeIndex>,
) -> Graph<SRef, EdgeWeight> {
    let mut graph = graph.clone();
    let remove_edges = edge_mapping
        .iter()
        .filter_map(|((_, w, _), index)| match w {
            EdgeWeight::Offset(o) if *o < 0 => Some(*index),
            _ => None,
        })
        .collect::<Vec<EdgeIndex>>();
    remove_edges.into_iter().for_each(|edge| {
        graph.remove_edge(edge);
    });
    graph
}

pub(crate) fn graph_without_self_filter_edges(
    graph: &Graph<SRef, EdgeWeight>,
    edge_mapping: &HashMap<(SRef, &EdgeWeight, SRef), EdgeIndex>,
) -> Graph<SRef, EdgeWeight> {
    let mut graph = graph.clone();
    let remove_edges = edge_mapping
        .iter()
        .filter_map(|((src, w, tar), index)| if src == tar { Some((w, index)) } else { None })
        .filter_map(|(w, index)| match w {
            EdgeWeight::Filter(w) if &EdgeWeight::Offset(0) == w.as_ref() => Some(*index),
            _ => None,
        })
        .collect::<Vec<EdgeIndex>>();
    remove_edges.into_iter().for_each(|edge| {
        graph.remove_edge(edge);
    });
    graph
}

pub(crate) fn graph_without_close_edges(
    graph: &Graph<SRef, EdgeWeight>,
    edge_mapping: &HashMap<(SRef, &EdgeWeight, SRef), EdgeIndex>,
) -> Graph<SRef, EdgeWeight> {
    let mut graph = graph.clone();
    let remove_edges = edge_mapping
        .iter()
        .filter_map(|((src, w, tar), index)| match (w, src == tar) {
            (EdgeWeight::Close(_), true) => Some(*index),
            _ => None,
        })
        .collect::<Vec<EdgeIndex>>();
    remove_edges.into_iter().for_each(|edge| {
        graph.remove_edge(edge);
    });
    graph
}
