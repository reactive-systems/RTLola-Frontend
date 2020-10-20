use super::EdgeWeight;
use super::SRef;
use petgraph::{graph::EdgeIndex, Graph};
use std::collections::HashMap;

pub(crate) fn graph_without_negative_offset_edges(
    graph: &Graph<SRef, EdgeWeight>,
    edge_mapping: &HashMap<(SRef, EdgeWeight, SRef), EdgeIndex>,
) -> Graph<SRef, EdgeWeight> {
    let mut graph = graph.clone();
    let remove_edges = edge_mapping
        .iter()
        .filter_map(|((_, weight, _), index)| match weight {
            EdgeWeight::Offset(o) => {
                if *o < 0 {
                    Some(*index)
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<EdgeIndex>>();
    remove_edges.into_iter().for_each(|edge| {
        graph.remove_edge(edge);
    });
    graph
}
