use crate::common_ir::{Layer, SRef, WRef};

use super::{Dependencies, DependencyGraph};

use std::collections::HashMap;
use std::convert::TryFrom;

use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::{expression::StreamAccessKind, Hir};
use crate::{
    common_ir::Offset,
    hir::modes::{ir_expr::WithIrExpr, HirMode},
};
use petgraph::{algo::is_cyclic_directed, graph::EdgeIndex, graph::NodeIndex, Graph};

pub(crate) trait DependenciesAnalyzed {
    // https://github.com/rust-lang/rust/issues/63063
    // type I1 = impl Iterator<Item = SRef>;
    // type I2 = impl Iterator<Item = (SRef, WRef)>;

    fn accesses(&self, who: SRef) -> Vec<SRef>;

    fn accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn layer(&self, sr: SRef) -> Layer;
}

impl DependenciesAnalyzed for Dependencies {
    fn accesses(&self, who: SRef) -> Vec<SRef> {
        self.dg
            .accesses
            .get(&who)
            .map_or(Vec::new(), |accesses| accesses.iter().map(|sref| *sref).collect::<Vec<SRef>>())
    }

    fn accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.dg
            .accessed_by
            .get(&who)
            .map_or(Vec::new(), |accessed_by| accessed_by.iter().map(|sref| *sref).collect::<Vec<SRef>>())
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.dg.aggregated_by.get(&who).map_or(Vec::new(), |aggregated_by| {
            aggregated_by.iter().map(|(sref, wref)| (*sref, *wref)).collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.dg.aggregates.get(&who).map_or(Vec::new(), |aggregates| {
            aggregates.iter().map(|(sref, wref)| (*sref, *wref)).collect::<Vec<(SRef, WRef)>>()
        })
    }

    fn layer(&self, _sr: SRef) -> Layer {
        todo!()
    }
}

pub(crate) trait DependenciesWrapper {
    type InnerD: DependenciesAnalyzed;
    fn inner_dep(&self) -> &Self::InnerD;
}

impl<A: DependenciesWrapper<InnerD = T>, T: DependenciesAnalyzed + 'static> DependenciesAnalyzed for A {
    fn accesses(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().accesses(who)
    }

    fn accessed_by(&self, who: SRef) -> Vec<SRef> {
        self.inner_dep().accessed_by(who)
    }

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregated_by(who)
    }

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)> {
        self.inner_dep().aggregates(who)
    }

    fn layer(&self, sr: SRef) -> Layer {
        self.inner_dep().layer(sr)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DependencyErr {
    NegativeCycle, // Should probably contain the cycle
    WellFormedNess,
}

pub(crate) struct DependencyReport {}

type Result<T> = std::result::Result<T, DependencyErr>;
type DG = Graph<SRef, EdgeWeight>;

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeWeight {
    Infinite,
    Offset(i32),
    Aggr(WRef),
    Hold,
}

// impl FloatMeasure for EdgeWeight {
//     fn zero() -> Self {
//         EdgeWeight::Offset(0)
//     }

//     fn infinite() -> Self {
//         EdgeWeight::Infinite
//     }
// }

// impl std::ops::Add for EdgeWeight {
//     type Output = EdgeWeight;

//     fn add(self, rhs: Self) -> Self::Output {
//         match (self, rhs) {
//             (EdgeWeight::Infinite, _) | (_, EdgeWeight::Infinite) => EdgeWeight::Infinite,
//             (EdgeWeight::Offset(w1), EdgeWeight::Offset(w2)) => EdgeWeight::Offset(w1 + w2),
//             (EdgeWeight::Offset(w), _) | (_, EdgeWeight::Offset(w)) => EdgeWeight::Offset(w),
//             (EdgeWeight::Aggr(wref), _) => EdgeWeight::Aggr(wref),
//             (EdgeWeight::Hold, EdgeWeight::Aggr(wref)) => EdgeWeight::Aggr(wref),
//             (EdgeWeight::Hold, EdgeWeight::Hold) => EdgeWeight::Hold,
//         }
//     }
// }

// impl PartialOrd for EdgeWeight {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         match (self, other) {
//             (EdgeWeight::Infinite, EdgeWeight::Infinite) => None,
//             (EdgeWeight::Infinite, _) => Some(std::cmp::Ordering::Greater),
//             (_, EdgeWeight::Infinite) => Some(std::cmp::Ordering::Less),
//             (EdgeWeight::Offset(w1), EdgeWeight::Offset(w2)) => w1.partial_cmp(w2),
//             (_, EdgeWeight::Offset(_)) => Some(std::cmp::Ordering::Less),
//             (EdgeWeight::Offset(_), _) => Some(std::cmp::Ordering::Greater),
//             (EdgeWeight::Aggr(_wref1), EdgeWeight::Aggr(_wref2)) => unimplemented!(), //Some(std::cmp::Ordering::Equal),
//             (EdgeWeight::Aggr(_wref), EdgeWeight::Hold) => Some(std::cmp::Ordering::Greater),
//             (EdgeWeight::Hold, EdgeWeight::Aggr(_)) => Some(std::cmp::Ordering::Less),
//             (EdgeWeight::Hold, EdgeWeight::Hold) => Some(std::cmp::Ordering::Equal),
//         }
//     }
// }

// impl Default for EdgeWeight {
//     fn default() -> Self {
//         EdgeWeight::Offset(0)
//     }
// }

impl Dependencies {
    pub(crate) fn analyze<M: HirMode + 'static + WithIrExpr>(spec: &Hir<M>) -> Result<Dependencies>
    where
        M: WithIrExpr + HirMode,
    {
        let num_nodes = spec.num_inputs() + spec.num_outputs() + spec.num_triggers();
        let num_edges = num_nodes; // Todo: improve estimate.
        let mut graph: DG = Graph::with_capacity(num_nodes, num_edges);
        let node_mapping: HashMap<SRef, NodeIndex> = spec.all_streams().map(|sr| (sr, graph.add_node(sr))).collect();
        let edges = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(sr, spec.expr(sr)))
            .map(|(src, w, tar)| {
                let weight = match w {
                    StreamAccessKind::Sync => EdgeWeight::Offset(0),
                    StreamAccessKind::Offset(o) => match o {
                        Offset::PastDiscreteOffset(o) => EdgeWeight::Offset(-i32::try_from(o).unwrap()),
                        Offset::FutureDiscreteOffset(o) => EdgeWeight::Offset(i32::try_from(o).unwrap()),
                        _ => todo!(),
                    },
                    StreamAccessKind::Hold => EdgeWeight::Hold,
                    StreamAccessKind::SlidingWindow(wref) => EdgeWeight::Aggr(wref),
                    StreamAccessKind::DiscreteWindow(wref) => EdgeWeight::Aggr(wref),
                };
                (src, weight, tar)
            })
            .collect::<Vec<(SRef, EdgeWeight, SRef)>>();
        let edge_mapping: HashMap<(SRef, EdgeWeight, SRef), EdgeIndex> = edges
            .iter()
            .map(|(src, w, tar)| ((*src, *w, *tar), graph.add_edge(node_mapping[src], node_mapping[tar], *w)))
            .collect();
        // Check well-formedness = no closed-walk with total weight of zero or positive
        Self::check_well_formedness(&graph, &edge_mapping)?;
        let mut accesses: HashMap<SRef, Vec<SRef>> = HashMap::new();
        let mut accessed_by: HashMap<SRef, Vec<SRef>> = HashMap::new();
        let mut aggregates: HashMap<SRef, Vec<(SRef, WRef)>> = HashMap::new();
        let mut aggregated_by: HashMap<SRef, Vec<(SRef, WRef)>> = HashMap::new();
        edges.iter().for_each(|(src, w, tar)| {
            (*accesses.entry(*src).or_insert(Vec::new())).push(*tar);
            (*accessed_by.entry(*tar).or_insert(Vec::new())).push(*src);
            if let EdgeWeight::Aggr(wref) = w {
                (*aggregates.entry(*src).or_insert(Vec::new())).push((*tar, *wref));
                (*aggregated_by.entry(*tar).or_insert(Vec::new())).push((*src, *wref));
            }
        });
        DependencyGraph { accesses, accessed_by, aggregates, aggregated_by };
        // Ok(Dependencies {  dg: graph })
        unimplemented!()
    }

    fn check_well_formedness(graph: &DG, edge_mapping: &HashMap<(SRef, EdgeWeight, SRef), EdgeIndex>) -> Result<()> {
        let mut graph = graph.clone(); // Clone graph to adapt edges.
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
        // check if cyclic
        if is_cyclic_directed(&graph) {
            Err(DependencyErr::WellFormedNess)
        } else {
            Ok(())
        }
    }

    fn collect_edges(src: SRef, expr: &Expression) -> Vec<(SRef, StreamAccessKind, SRef)> {
        match &expr.kind {
            ExpressionKind::StreamAccess(target, stream_access_kind) => vec![(src, *stream_access_kind, *target)],
            ExpressionKind::LoadConstant(_) => Vec::new(),
            ExpressionKind::ArithLog(_op, args) => {
                args.iter().flat_map(|a| Self::collect_edges(src, a).into_iter()).collect()
            }
            ExpressionKind::Tuple(content) => content.iter().flat_map(|a| Self::collect_edges(src, a)).collect(),
            ExpressionKind::Function(_name, args) => args.iter().flat_map(|a| Self::collect_edges(src, a)).collect(),
            ExpressionKind::Ite { condition, consequence, alternative } => Self::collect_edges(src, condition)
                .into_iter()
                .chain(Self::collect_edges(src, consequence).into_iter())
                .chain(Self::collect_edges(src, alternative).into_iter())
                .collect(),
            ExpressionKind::TupleAccess(content, _n) => Self::collect_edges(src, content),
            ExpressionKind::Widen(inner) => Self::collect_edges(src, inner),
            ExpressionKind::Default { expr, default } => Self::collect_edges(src, expr)
                .into_iter()
                .chain(Self::collect_edges(src, default).into_iter())
                .collect(),
        }
    }

    fn offset_to_weight(_o: &Offset) -> u32 {
        todo!()
    }
}
