use std::collections::HashMap;
use std::convert::TryFrom;

use petgraph::algo::{has_path_connecting, is_cyclic_directed};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::Outgoing;

use super::{DepAna, DepAnaTrait, TypedTrait};
use crate::hir::{Expression, ExpressionKind, FnExprKind, Hir, Offset, SRef, StreamAccessKind, WRef, WidenExprKind};
use crate::modes::HirMode;

/// Represents the Dependency Graph
///
/// The dependency graph represents all dependecies between streams.
/// For this, the graph contains a node for each node in the specification and an edge from `source` to `target`, iff the stream `source` uses an stream value of `target`.
/// The weight of each nodes is the stream reference representing the stream. The weight of the edges is the [kind](EdgeWeight) of the lookup.
pub type DependencyGraph = StableGraph<SRef, EdgeWeight>;
/// Represents the weights of the edges in the dependency graph
#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub enum EdgeWeight {
    /// Offset weigth
    Offset(i32),
    /// Window weigth
    Aggr(WRef),
    /// Sample and hold weigth
    Hold,
    /// Represents the edge weight of lookups appearing in the spawn condition
    Spawn(Box<EdgeWeight>),
    /// Represents the edge weight of lookups appearing in the filter condition
    Filter(Box<EdgeWeight>),
    /// Represents the edge weight of lookups appearing in the close condition
    Close(Box<EdgeWeight>),
}

/// Represents all dependencies between streams
pub(crate) type Streamdependencies = HashMap<SRef, Vec<SRef>>;
/// Represents all dependencies between streams in which a window lookup is used
pub(crate) type Windowdependencies = HashMap<SRef, Vec<(SRef, WRef)>>;

pub(crate) trait ExtendedDepGraph {
    /// Returns a new [dependency graph](DependencyGraph), in which all edges representing a negative offset lookup are deleted
    fn without_negative_offset_edges(&self) -> Self;

    /// Returns `true`, iff the edge weight `e` is a negative offset lookup
    fn has_negative_offset(e: &EdgeWeight) -> bool {
        match e {
            EdgeWeight::Offset(o) if *o < 0 => true,
            EdgeWeight::Spawn(s) => Self::has_negative_offset(s),
            EdgeWeight::Filter(f) => Self::has_negative_offset(f),
            EdgeWeight::Close(c) => Self::has_negative_offset(c),
            _ => false,
        }
    }

    /// Returns a new [dependency graph](DependencyGraph), in which all edges representing a lookup that are used in the close condition are deleted
    fn without_close_edges(&self) -> Self;

    /// Returns a new [dependency graph](DependencyGraph), which only contains edges representing a lookup in the spawn condition
    fn only_spawn_edges(&self) -> Self;

    /// Returns two new [dependency graphs](DependencyGraph), one containing the streams with an event-based pacing type and one with a periodic pacing type
    ///
    /// This function returns two new [dependency graphs](DependencyGraph):
    /// The first graph consists of all streams with an event-based pacing type. Additionally, it only contains the edges between two event-based streams.
    /// The second graph consists of all streams with a periodic pacing type. Additionally, it only contains the edges between two periodic streams.
    fn split_graph<M>(&self, spec: &Hir<M>) -> (Self, Self)
    where
        M: HirMode + DepAnaTrait + TypedTrait,
        Self: Sized;
}

impl ExtendedDepGraph for DependencyGraph {
    fn without_negative_offset_edges(&self) -> Self {
        let mut working_graph = self.clone();
        self.edge_indices().for_each(|edge_index| {
            if Self::has_negative_offset(self.edge_weight(edge_index).unwrap()) {
                working_graph.remove_edge(edge_index);
            }
        });
        working_graph
    }

    fn without_close_edges(&self) -> Self {
        let mut working_graph = self.clone();
        self.edge_indices().for_each(|edge_index| {
            if let EdgeWeight::Close(_) = self.edge_weight(edge_index).unwrap() {
                working_graph.remove_edge(edge_index);
            }
        });
        working_graph
    }

    fn only_spawn_edges(&self) -> Self {
        let mut working_graph = self.clone();
        self.edge_indices().for_each(|edge_index| {
            if let EdgeWeight::Spawn(_) = self.edge_weight(edge_index).unwrap() {
            } else {
                let res = working_graph.remove_edge(edge_index);
                assert!(res.is_some());
            }
        });
        working_graph
    }

    fn split_graph<M>(&self, spec: &Hir<M>) -> (Self, Self)
    where
        M: HirMode + DepAnaTrait + TypedTrait,
        Self: Sized,
    {
        // remove edges and nodes, so mapping does not change
        let mut event_graph = self.clone();
        let mut periodic_graph = self.clone();
        self.edge_indices().for_each(|edge_index| {
            let (src, tar) = self.edge_endpoints(edge_index).unwrap();
            let src = self.node_weight(src).unwrap();
            let tar = self.node_weight(tar).unwrap();
            match (spec.is_event(*src), spec.is_event(*tar)) {
                (true, true) => {
                    periodic_graph.remove_edge(edge_index);
                },
                (false, false) => {
                    event_graph.remove_edge(edge_index);
                },
                _ => {
                    event_graph.remove_edge(edge_index);
                    periodic_graph.remove_edge(edge_index);
                },
            }
        });
        // delete nodes
        self.node_indices().for_each(|node_index| {
            let node_sref = self.node_weight(node_index).unwrap();
            if spec.is_event(*node_sref) {
                periodic_graph.remove_node(node_index);
            } else {
                assert!(spec.is_periodic(*node_sref));
                event_graph.remove_node(node_index);
            }
        });
        (event_graph, periodic_graph)
    }
}

impl DepAnaTrait for DepAna {
    fn direct_accesses(&self, who: SRef) -> Vec<SRef> {
        self.direct_accesses
            .get(&who)
            .map_or(Vec::new(), |accesses| accesses.iter().copied().collect::<Vec<SRef>>())
    }

    fn transitive_accesses(&self, who: SRef) -> Vec<SRef> {
        self.transitive_accesses
            .get(&who)
            .map_or(Vec::new(), |accesses| accesses.iter().copied().collect::<Vec<SRef>>())
    }

    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.direct_accessed_by.get(&who).map_or(Vec::new(), |accessed_by| {
            accessed_by.iter().copied().collect::<Vec<SRef>>()
        })
    }

    fn transitive_accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.transitive_accessed_by
            .get(&who)
            .map_or(Vec::new(), |accesses| accesses.iter().copied().collect::<Vec<SRef>>())
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.aggregated_by.get(&who).map_or(Vec::new(), |aggregated_by| {
            aggregated_by
                .iter()
                .map(|(sref, wref)| (*sref, *wref))
                .collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.aggregates.get(&who).map_or(Vec::new(), |aggregates| {
            aggregates
                .iter()
                .map(|(sref, wref)| (*sref, *wref))
                .collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn graph(&self) -> &DependencyGraph {
        &self.graph
    }
}

/// Represents the error of the dependency analysis
#[derive(Debug, Clone, Copy)]
pub enum DependencyErr {
    /// Represents the error that the well-formedness condition is not satisfied.
    ///
    /// This error indicates that the given specification is not well-formed, i.e., the dependency graph contains a non-negative cycle.
    WellFormedNess,
}

type Result<T> = std::result::Result<T, DependencyErr>;

impl DepAna {
    /// Returns the result of the dependency analysis
    ///
    /// This function analyzes the dependencies of the given `spec`. It returns an [DependencyErr] if the specification is not well-formed.
    /// Otherwise, the function returns the dependencies in the specification, including the dependency graph.
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Result<DepAna>
    where
        M: HirMode,
    {
        let num_nodes = spec.num_inputs() + spec.num_outputs() + spec.num_triggers();
        let num_edges = num_nodes; // Todo: improve estimate.
        let mut graph: DependencyGraph = StableGraph::with_capacity(num_nodes, num_edges);
        let edges_expr = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(spec, sr, spec.expr(sr)))
            .map(|(src, w, tar)| (src, Self::stream_access_kind_to_edge_weight(w), tar));
        let edges_spawn = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| {
                spec.spawn(sr).map(|(spawn_expr, spawn_cond)| {
                    spawn_expr
                        .map_or(Vec::new(), |spawn_expr| Self::collect_edges(spec, sr, spawn_expr))
                        .into_iter()
                        .chain(spawn_cond.map_or(Vec::new(), |spawn_cond| Self::collect_edges(spec, sr, spawn_cond)))
                })
            })
            .flatten()
            .map(|(src, w, tar)| {
                (
                    src,
                    EdgeWeight::Spawn(Box::new(Self::stream_access_kind_to_edge_weight(w))),
                    tar,
                )
            });
        let edges_filter = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| spec.filter(sr).map(|filter| Self::collect_edges(spec, sr, filter)))
            .flatten()
            .map(|(src, w, tar)| {
                (
                    src,
                    EdgeWeight::Filter(Box::new(Self::stream_access_kind_to_edge_weight(w))),
                    tar,
                )
            });
        let edges_close = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| spec.close(sr).map(|close| Self::collect_edges(spec, sr, close)))
            .flatten()
            .map(|(src, w, tar)| {
                (
                    src,
                    EdgeWeight::Close(Box::new(Self::stream_access_kind_to_edge_weight(w))),
                    tar,
                )
            });
        let edges = edges_expr
            .chain(edges_spawn)
            .chain(edges_filter)
            .chain(edges_close)
            .collect::<Vec<(SRef, EdgeWeight, SRef)>>(); // TODO can use this approxiamtion for the number of edges

        // add nodes and edges to graph
        let node_mapping: HashMap<SRef, NodeIndex> = spec.all_streams().map(|sr| (sr, graph.add_node(sr))).collect();
        edges.iter().for_each(|(src, w, tar)| {
            graph.add_edge(node_mapping[src], node_mapping[tar], w.clone());
        });

        // Check well-formedness = no closed-walk with total weight of zero or positive
        Self::check_well_formedness(&graph)?;
        // Describe dependencies in HashMaps
        let mut direct_accesses: HashMap<SRef, Vec<SRef>> = spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        let mut direct_accessed_by: HashMap<SRef, Vec<SRef>> = spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        let mut aggregates: HashMap<SRef, Vec<(SRef, WRef)>> = spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        let mut aggregated_by: HashMap<SRef, Vec<(SRef, WRef)>> =
            spec.all_streams().map(|sr| (sr, Vec::new())).collect();
        edges.iter().for_each(|(src, w, tar)| {
            let cur_accesses = direct_accesses.get_mut(src).unwrap();
            if !cur_accesses.contains(tar) {
                cur_accesses.push(*tar);
            }
            let cur_accessed_by = direct_accessed_by.get_mut(tar).unwrap();
            if !cur_accessed_by.contains(src) {
                cur_accessed_by.push(*src);
            }
            if let EdgeWeight::Aggr(wref) = w {
                let cur_aggregates = aggregates.get_mut(src).unwrap();
                if !cur_aggregates.contains(&(*tar, *wref)) {
                    cur_aggregates.push((*tar, *wref));
                }
                let cur_aggregates_by = aggregated_by.get_mut(tar).unwrap();
                if !cur_aggregates_by.contains(&(*src, *wref)) {
                    cur_aggregates_by.push((*src, *wref));
                }
            }
        });
        let transitive_accesses = graph
            .node_indices()
            .map(|from_index| {
                let sr = *(graph.node_weight(from_index).unwrap());
                let transitive_dependencies = graph
                    .node_indices()
                    .filter(|to_index| Self::has_transitive_connection(&graph, from_index, *to_index))
                    .map(|to_index| *(graph.node_weight(to_index).unwrap()))
                    .collect::<Vec<SRef>>();
                (sr, transitive_dependencies)
            })
            .collect::<HashMap<SRef, Vec<SRef>>>();
        let transitive_accessed_by = graph
            .node_indices()
            .map(|to_index| {
                let sr = *(graph.node_weight(to_index).unwrap());
                let transitive_dependencies = graph
                    .node_indices()
                    .filter(|from_index| Self::has_transitive_connection(&graph, *from_index, to_index))
                    .map(|from_index| *(graph.node_weight(from_index).unwrap()))
                    .collect::<Vec<SRef>>();
                (sr, transitive_dependencies)
            })
            .collect::<HashMap<SRef, Vec<SRef>>>();
        Ok(DepAna {
            direct_accesses,
            transitive_accesses,
            direct_accessed_by,
            transitive_accessed_by,
            aggregated_by,
            aggregates,
            graph,
        })
    }

    fn has_transitive_connection(graph: &DependencyGraph, from: NodeIndex, to: NodeIndex) -> bool {
        if from != to {
            has_path_connecting(&graph, from, to, None)
        } else {
            // Seprate case: has_path_connection alwas return true, if from = to
            // First check if self edge is contained
            let self_check = graph
                .edge_indices()
                .filter(|ei| {
                    let (src, tar) = graph.edge_endpoints(*ei).unwrap();
                    src == tar
                }) // filter self references
                .find(|ei| {
                    let (src, _) = graph.edge_endpoints(*ei).unwrap();
                    src == from
                });
            // Second check if one of the neighbors has a connection to the current node
            let neighbor_check = graph
                .neighbors_directed(from, Outgoing)
                .any(|cur_neigbor| has_path_connecting(graph, cur_neigbor, to, None));
            self_check.is_some() || neighbor_check
        }
    }

    /// Returns is the DP is well-formed, i.e., no closed-walk with total weight of zero or positive
    fn check_well_formedness(graph: &DependencyGraph) -> Result<()> {
        let graph = graph.without_negative_offset_edges();
        let graph = graph.without_close_edges();
        // check if cyclic
        if is_cyclic_directed(&graph) {
            Err(DependencyErr::WellFormedNess)
        } else {
            Ok(())
        }
    }

    fn collect_edges<M>(spec: &Hir<M>, src: SRef, expr: &Expression) -> Vec<(SRef, StreamAccessKind, SRef)>
    where
        M: HirMode,
    {
        match &expr.kind {
            ExpressionKind::StreamAccess(target, stream_access_kind, args) => {
                let mut args = args
                    .iter()
                    .map(|arg| Self::collect_edges(spec, src, arg))
                    .flatten()
                    .collect::<Vec<(SRef, StreamAccessKind, SRef)>>();
                args.push((src, *stream_access_kind, *target));
                args
            },
            ExpressionKind::ParameterAccess(_, _) => Vec::new(),
            ExpressionKind::LoadConstant(_) => Vec::new(),
            ExpressionKind::ArithLog(_op, args) => {
                args.iter()
                    .flat_map(|a| Self::collect_edges(spec, src, a).into_iter())
                    .collect()
            },
            ExpressionKind::Tuple(content) => content.iter().flat_map(|a| Self::collect_edges(spec, src, a)).collect(),
            ExpressionKind::Function(FnExprKind { args, .. }) => {
                args.iter().flat_map(|a| Self::collect_edges(spec, src, a)).collect()
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                Self::collect_edges(spec, src, condition)
                    .into_iter()
                    .chain(Self::collect_edges(spec, src, consequence).into_iter())
                    .chain(Self::collect_edges(spec, src, alternative).into_iter())
                    .collect()
            },
            ExpressionKind::TupleAccess(content, _n) => Self::collect_edges(spec, src, content),
            ExpressionKind::Widen(WidenExprKind { expr: inner, .. }) => Self::collect_edges(spec, src, inner),
            ExpressionKind::Default { expr, default } => {
                Self::collect_edges(spec, src, expr)
                    .into_iter()
                    .chain(Self::collect_edges(spec, src, default).into_iter())
                    .collect()
            },
        }
    }

    fn stream_access_kind_to_edge_weight(w: StreamAccessKind) -> EdgeWeight {
        match w {
            StreamAccessKind::Sync => EdgeWeight::Offset(0),
            StreamAccessKind::Offset(o) => {
                match o {
                    Offset::PastDiscrete(o) => EdgeWeight::Offset(-i32::try_from(o).unwrap()),
                    Offset::FutureDiscrete(o) => {
                        if o == 0 {
                            EdgeWeight::Offset(i32::try_from(o).unwrap())
                        } else {
                            todo!("implement dependency analysis for positive future offsets")
                        }
                    },
                    _ => todo!("implement dependency analysis for real-time offsets"),
                }
            },
            StreamAccessKind::Hold => EdgeWeight::Hold,

            StreamAccessKind::SlidingWindow(wref) => EdgeWeight::Aggr(wref),
            StreamAccessKind::DiscreteWindow(wref) => EdgeWeight::Aggr(wref),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rtlola_parser::{parse_with_handler, ParserConfig};
    use rtlola_reporting::Handler;

    use super::*;
    use crate::modes::BaseMode;

    fn check_graph_for_spec(
        spec: &str,
        dependencies: Option<(
            HashMap<SRef, Vec<SRef>>,
            HashMap<SRef, Vec<SRef>>,
            HashMap<SRef, Vec<SRef>>,
            HashMap<SRef, Vec<SRef>>,
            HashMap<SRef, Vec<(SRef, WRef)>>,
            HashMap<SRef, Vec<(SRef, WRef)>>,
        )>,
    ) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = parse_with_handler(ParserConfig::for_string(spec.to_string()), &handler)
            .unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<BaseMode>::from_ast(ast, &handler).unwrap();
        let deps = DepAna::analyze(&hir);
        if let Ok(deps) = deps {
            let (
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            ) = dependencies.unwrap();
            deps.direct_accesses.iter().for_each(|(sr, accesses_hir)| {
                let accesses_reference = direct_accesses.get(sr).unwrap();
                assert_eq!(accesses_hir.len(), accesses_reference.len(), "sr: {}", sr);
                accesses_hir
                    .iter()
                    .for_each(|sr| assert!(accesses_reference.contains(sr), "sr: {}", sr));
            });
            deps.transitive_accesses.iter().for_each(|(sr, accesses_hir)| {
                let accesses_reference = transitive_accesses.get(sr).unwrap();
                assert_eq!(accesses_hir.len(), accesses_reference.len(), "sr: {}", sr);
                accesses_hir
                    .iter()
                    .for_each(|sr| assert!(accesses_reference.contains(sr), "sr: {}", sr));
            });
            deps.direct_accessed_by.iter().for_each(|(sr, accessed_by_hir)| {
                let accessed_by_reference = direct_accessed_by.get(sr).unwrap();
                assert_eq!(accessed_by_hir.len(), accessed_by_reference.len(), "sr: {}", sr);
                accessed_by_hir
                    .iter()
                    .for_each(|sr| assert!(accessed_by_reference.contains(sr), "sr: {}", sr));
            });
            deps.transitive_accessed_by.iter().for_each(|(sr, accessed_by_hir)| {
                let accessed_by_reference = transitive_accessed_by.get(sr).unwrap();
                assert_eq!(accessed_by_hir.len(), accessed_by_reference.len(), "sr: {}", sr);
                accessed_by_hir
                    .iter()
                    .for_each(|sr| assert!(accessed_by_reference.contains(sr), "sr: {}", sr));
            });
            deps.aggregates.iter().for_each(|(sr, aggregates_hir)| {
                let aggregates_reference = aggregates.get(sr).unwrap();
                assert_eq!(aggregates_hir.len(), aggregates_reference.len(), "test");
                aggregates_hir
                    .iter()
                    .for_each(|lookup| assert!(aggregates_reference.contains(lookup)));
            });
            deps.aggregated_by.iter().for_each(|(sr, aggregated_by_hir)| {
                let aggregated_by_reference = aggregated_by.get(sr).unwrap();
                assert_eq!(aggregated_by_hir.len(), aggregated_by_reference.len(), "test");
                aggregated_by_hir
                    .iter()
                    .for_each(|lookup| assert!(aggregated_by_reference.contains(lookup)));
            });
        } else {
            assert!(dependencies.is_none())
        }
    }

    #[test]
    fn self_loop_simple() {
        let spec = "input a: Int8\noutput b: Int8 := a+b";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn self_loop_complex() {
        let spec = "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c\noutput e: Int8 := e";
        check_graph_for_spec(spec, None)
    }

    #[test]
    fn simple_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a+d\noutput c: Int8 := b\noutput d: Int8 := c";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn linear_dependencies() {
        let spec = "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"]]),
            (sname_to_sref["c"], vec![sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"]]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["c"]],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (sname_to_sref["b"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn negative_loop() {
        let spec = "output a: Int8 := a.offset(by: -1).defaults(to: 0)";
        let sname_to_sref = vec![("a", SRef::Out(0))].into_iter().collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![(sname_to_sref["a"], vec![sname_to_sref["a"]])]
            .into_iter()
            .collect();
        let transitive_accesses = vec![(sname_to_sref["a"], vec![sname_to_sref["a"]])]
            .into_iter()
            .collect();
        let direct_accessed_by = vec![(sname_to_sref["a"], vec![sname_to_sref["a"]])]
            .into_iter()
            .collect();
        let transitive_accessed_by = vec![(sname_to_sref["a"], vec![sname_to_sref["a"]])]
            .into_iter()
            .collect();
        let aggregates = vec![(sname_to_sref["a"], vec![])].into_iter().collect();
        let aggregated_by = vec![(sname_to_sref["a"], vec![])].into_iter().collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn negative_loop_different_offsets() {
        let spec = "input a: Int8\noutput b: Int8 := a.offset(by: -1).defaults(to: 0) + d.offset(by:-2).defaults(to:0)\noutput c: Int8 := b.offset(by:-3).defaults(to: 0)\noutput d: Int8 := c.offset(by:-4).defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (
                sname_to_sref["b"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
            (
                sname_to_sref["c"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
            (
                sname_to_sref["d"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["b"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["c"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn negative_and_postive_lookups_as_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a + d.offset(by:-1).defaults(to:0)\noutput c: Int8 := b\noutput d: Int8 := c";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (
                sname_to_sref["b"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
            (
                sname_to_sref["c"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
            (
                sname_to_sref["d"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["b"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["c"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn self_sliding_window() {
        let spec = "output a := a.aggregate(over: 1s, using: sum)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn sliding_window_loop() {
        let spec = "input a: Int8\noutput b@10Hz := a.aggregate(over: 1s, using: sum) + d.aggregate(over: 1s, using: sum)\noutput c@2Hz := b.aggregate(over: 1s, using: sum)\noutput d@1Hz := c.aggregate(over: 1s, using: sum)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn sliding_window_and_positive_lookups_loop() {
        let spec = "input a: Int8\noutput b@10Hz := a.aggregate(over: 1s, using: sum) + d.hold().defaults(to: 0)\noutput c@2Hz := b.aggregate(over: 1s, using: sum)\noutput d@1Hz := c";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn sliding_windows_chain_and_hold_lookup() {
        let spec = "input a: Int8\noutput b@1Hz := a.aggregate(over: 1s, using: sum) + d.offset(by: -1).defaults(to: 0)\noutput c@2Hz := b.aggregate(over: 1s, using: sum)\noutput d@2Hz := b.hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (
                sname_to_sref["b"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["c"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["d"]],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (
                sname_to_sref["b"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
            (sname_to_sref["c"], vec![]),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["b"], sname_to_sref["c"], sname_to_sref["d"]],
            ),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![(sname_to_sref["a"], WRef::Sliding(0))]),
            (sname_to_sref["c"], vec![(sname_to_sref["b"], WRef::Sliding(1))]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![(sname_to_sref["b"], WRef::Sliding(0))]),
            (sname_to_sref["b"], vec![(sname_to_sref["c"], WRef::Sliding(1))]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    // #[test]
    // #[ignore] // Graph Analysis cannot handle positive edges; not required for this branch.
    // fn positive_loop_should_cause_a_warning() {
    //     chek_graph_for_spec("output a: Int8 := a[1]", false);
    // }

    #[test]
    fn parallel_edges_in_a_loop() {
        let spec = "input a: Int8\noutput b: Int8 := a+d+d\noutput c: Int8 := b\noutput d: Int8 := c";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn spawn_self_loop() {
        let spec = "input a: Int8\noutput b spawn if b > 6 := a + 5";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn spawn_self_negative_loop() {
        let spec = "input a: Int8\noutput b spawn if b.offset(by: -1).defaults(to: 0) > 6 := a + 5";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![(sname_to_sref["a"], vec![]), (sname_to_sref["b"], vec![])]
            .into_iter()
            .collect();
        let aggregated_by = vec![(sname_to_sref["a"], vec![]), (sname_to_sref["b"], vec![])]
            .into_iter()
            .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }
    #[test]
    fn filter_self_loop() {
        let spec = "input a: Int8\noutput b filter b > 6 := a + 5";
        check_graph_for_spec(spec, None);
    }
    #[test]
    fn close_self_loop() {
        let spec = "input a: Int8\noutput b close b > 6 := a + 5";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![(sname_to_sref["a"], vec![]), (sname_to_sref["b"], vec![])]
            .into_iter()
            .collect();
        let aggregated_by = vec![(sname_to_sref["a"], vec![]), (sname_to_sref["b"], vec![])]
            .into_iter()
            .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn simple_loop_with_parameter() {
        let spec = "input a: Int8\noutput b(para) spawn with c := b(para).offset(by: -1).defaults(to: 0)\noutput c := b(5).hold().defaults(to: 0)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn simple_chain_with_parameter() {
        let spec = "input a: Int8\noutput b := a + 5\noutput c(para) spawn with b := para + 5";
        let sname_to_sref = vec![("a", SRef::In(0)), ("b", SRef::Out(0)), ("c", SRef::Out(1))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"]]),
            (sname_to_sref["c"], vec![sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![sname_to_sref["a"]]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"]]),
            (sname_to_sref["c"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["b"], sname_to_sref["c"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"]]),
            (sname_to_sref["c"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn parameter_loop() {
        let spec = "input a: Int8\noutput b(para) spawn with a if d(para) > 6 := a + para\noutput c(para) spawn with a if a > 6 := a + b(para)\noutput d(para) spawn with a if a > 6 := a + c(para)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn lookup_chain_with_parametrization() {
        let spec = "input a: Int8\noutput b(para) spawn with a if a > 6 := a + para\noutput c(para) spawn with a if a > 6 := a + b(para)\noutput d(para) spawn with a if a > 6 := a + c(para)";
        let name_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::Out(0)),
            ("c", SRef::Out(1)),
            ("d", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (name_to_sref["a"], vec![]),
            (name_to_sref["b"], vec![name_to_sref["a"]]),
            (name_to_sref["c"], vec![name_to_sref["a"], name_to_sref["b"]]),
            (name_to_sref["d"], vec![name_to_sref["a"], name_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (name_to_sref["a"], vec![]),
            (name_to_sref["b"], vec![name_to_sref["a"]]),
            (name_to_sref["c"], vec![name_to_sref["a"], name_to_sref["b"]]),
            (
                name_to_sref["d"],
                vec![name_to_sref["a"], name_to_sref["b"], name_to_sref["c"]],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (
                name_to_sref["a"],
                vec![name_to_sref["b"], name_to_sref["c"], name_to_sref["d"]],
            ),
            (name_to_sref["b"], vec![name_to_sref["c"]]),
            (name_to_sref["c"], vec![name_to_sref["d"]]),
            (name_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                name_to_sref["a"],
                vec![name_to_sref["b"], name_to_sref["c"], name_to_sref["d"]],
            ),
            (name_to_sref["b"], vec![name_to_sref["c"], name_to_sref["d"]]),
            (name_to_sref["c"], vec![name_to_sref["d"]]),
            (name_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (name_to_sref["a"], vec![]),
            (name_to_sref["b"], vec![]),
            (name_to_sref["c"], vec![]),
            (name_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (name_to_sref["a"], vec![]),
            (name_to_sref["b"], vec![]),
            (name_to_sref["c"], vec![]),
            (name_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn parameter_loop_with_different_lookup_types() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a if a < b := p + b + f(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) := b + 5\noutput e(p) spawn with b := d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn parameter_loop_with_lookup_in_close() {
        let spec = "input a: Int8\ninput b: Int8\noutput c(p) spawn with a if a < b := p + b + g(p).hold().defaults(to: 0)\noutput d(p) spawn with b if c(4).hold().defaults(to: 0) := b + 5\noutput e(p) spawn with b := d(p).hold().defaults(to: 0) + 5\noutput f(p) spawn with b filter e(p).hold().defaults(to: 0) < 6 := b + 5\noutput g(p) spawn with b close f(p).hold().defaults(to: 0) < 6 := b + 5";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
            ("f", SRef::Out(3)),
            ("g", SRef::Out(4)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (
                sname_to_sref["c"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["g"]],
            ),
            (sname_to_sref["d"], vec![sname_to_sref["b"], sname_to_sref["c"]]),
            (sname_to_sref["e"], vec![sname_to_sref["b"], sname_to_sref["d"]]),
            (sname_to_sref["f"], vec![sname_to_sref["b"], sname_to_sref["e"]]),
            (sname_to_sref["g"], vec![sname_to_sref["b"], sname_to_sref["f"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (
                sname_to_sref["c"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["d"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["e"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["f"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["g"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["c"]]),
            (
                sname_to_sref["b"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![sname_to_sref["e"]]),
            (sname_to_sref["e"], vec![sname_to_sref["f"]]),
            (sname_to_sref["f"], vec![sname_to_sref["g"]]),
            (sname_to_sref["g"], vec![sname_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["b"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["c"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["d"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["e"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["f"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
            (
                sname_to_sref["g"],
                vec![
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                    sname_to_sref["e"],
                    sname_to_sref["f"],
                    sname_to_sref["g"],
                ],
            ),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
            (sname_to_sref["e"], vec![]),
            (sname_to_sref["f"], vec![]),
            (sname_to_sref["g"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
            (sname_to_sref["e"], vec![]),
            (sname_to_sref["f"], vec![]),
            (sname_to_sref["g"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn parameter_nested_lookup_implicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(c(b).hold().defaults(to: 0)).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["b"], sname_to_sref["c"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["c"]],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["c"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"]]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn parameter_nested_lookup_explicit() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let sname_to_sref = vec![
            ("a", SRef::In(0)),
            ("b", SRef::In(1)),
            ("c", SRef::Out(0)),
            ("d", SRef::Out(1)),
            ("e", SRef::Out(2)),
        ]
        .into_iter()
        .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
            (sname_to_sref["d"], vec![sname_to_sref["b"], sname_to_sref["c"]]),
            (sname_to_sref["e"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![sname_to_sref["a"], sname_to_sref["b"]]),
            (
                sname_to_sref["d"],
                vec![sname_to_sref["a"], sname_to_sref["b"], sname_to_sref["c"]],
            ),
            (
                sname_to_sref["e"],
                vec![
                    sname_to_sref["a"],
                    sname_to_sref["b"],
                    sname_to_sref["c"],
                    sname_to_sref["d"],
                ],
            ),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["c"]]),
            (sname_to_sref["b"], vec![sname_to_sref["c"], sname_to_sref["d"]]),
            (sname_to_sref["c"], vec![sname_to_sref["d"], sname_to_sref["e"]]),
            (sname_to_sref["d"], vec![sname_to_sref["e"]]),
            (sname_to_sref["e"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (
                sname_to_sref["a"],
                vec![sname_to_sref["c"], sname_to_sref["d"], sname_to_sref["e"]],
            ),
            (
                sname_to_sref["b"],
                vec![sname_to_sref["c"], sname_to_sref["d"], sname_to_sref["e"]],
            ),
            (sname_to_sref["c"], vec![sname_to_sref["d"], sname_to_sref["e"]]),
            (sname_to_sref["d"], vec![sname_to_sref["e"]]),
            (sname_to_sref["e"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
            (sname_to_sref["e"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["b"], vec![]),
            (sname_to_sref["c"], vec![]),
            (sname_to_sref["d"], vec![]),
            (sname_to_sref["e"], vec![]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }

    #[test]
    fn parameter_loop_with_parameter_lookup() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b + e\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn parameter_cross_lookup() {
        let spec = "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(e).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        check_graph_for_spec(spec, None);
    }

    #[test]
    fn delay() {
        let spec = "
            input x:Int8\n\
            output a @1Hz spawn if x=42 close if true then true else a := x.aggregate(over: 1s, using: sum) > 1337
        ";
        let sname_to_sref = vec![("a", SRef::Out(0)), ("x", SRef::In(0))]
            .into_iter()
            .collect::<HashMap<&str, SRef>>();
        let direct_accesses = vec![
            (sname_to_sref["a"], vec![sname_to_sref["a"], sname_to_sref["x"]]),
            (sname_to_sref["x"], vec![]),
        ]
        .into_iter()
        .collect();
        let transitive_accesses = vec![
            (sname_to_sref["a"], vec![sname_to_sref["a"], sname_to_sref["x"]]),
            (sname_to_sref["x"], vec![]),
        ]
        .into_iter()
        .collect();
        let direct_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["a"]]),
            (sname_to_sref["x"], vec![sname_to_sref["a"]]),
        ]
        .into_iter()
        .collect();
        let transitive_accessed_by = vec![
            (sname_to_sref["a"], vec![sname_to_sref["a"]]),
            (sname_to_sref["x"], vec![sname_to_sref["a"]]),
        ]
        .into_iter()
        .collect();
        let aggregates = vec![
            (sname_to_sref["a"], vec![(sname_to_sref["x"], WRef::Sliding(0))]),
            (sname_to_sref["x"], vec![]),
        ]
        .into_iter()
        .collect();
        let aggregated_by = vec![
            (sname_to_sref["a"], vec![]),
            (sname_to_sref["x"], vec![(sname_to_sref["a"], WRef::Sliding(0))]),
        ]
        .into_iter()
        .collect();
        check_graph_for_spec(
            spec,
            Some((
                direct_accesses,
                transitive_accesses,
                direct_accessed_by,
                transitive_accessed_by,
                aggregates,
                aggregated_by,
            )),
        );
    }
}
