use crate::common_ir::{SRef, WRef};

use super::{Dependencies, DependencyGraph, EdgeWeight};

use std::collections::HashMap;
use std::convert::TryFrom;

use super::dg_functionality::*;
use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::{expression::StreamAccessKind, Hir};
use crate::{
    common_ir::Offset,
    hir::modes::{ir_expr::WithIrExpr, HirMode},
};
use petgraph::{algo::is_cyclic_directed, Graph};

pub(crate) trait DependenciesAnalyzed {
    // https://github.com/rust-lang/rust/issues/63063
    // type I1 = impl Iterator<Item = SRef>;
    // type I2 = impl Iterator<Item = (SRef, WRef)>;

    fn accesses(&self, who: SRef) -> Vec<SRef>;

    fn accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>;

    fn graph(&self) -> &Graph<SRef, EdgeWeight>;
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

    fn graph(&self) -> &Graph<SRef, EdgeWeight> {
        &self.dg.graph
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

    fn graph(&self) -> &Graph<SRef, EdgeWeight> {
        self.inner_dep().graph()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DependencyErr {
    NegativeCycle, // Should probably contain the cycle
    WellFormedNess,
}

pub(crate) struct DependencyReport {}

type Result<T> = std::result::Result<T, DependencyErr>;

impl Dependencies {
    pub(crate) fn analyze<M>(spec: &Hir<M>) -> Result<Dependencies>
    where
        M: WithIrExpr + HirMode + 'static,
    {
        let num_nodes = spec.num_inputs() + spec.num_outputs() + spec.num_triggers();
        let num_edges = num_nodes; // Todo: improve estimate.
        let graph: Graph<SRef, EdgeWeight> = Graph::with_capacity(num_nodes, num_edges);
        let edges_expr = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(sr, spec.expr(sr)))
            .map(|(src, w, tar)| (src, Self::stream_access_kind_to_edge_weight(w), tar));
        let edges_spawn = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| {
                Self::collect_edges(sr, spec.spawn(sr).0).into_iter().chain(Self::collect_edges(sr, spec.spawn(sr).1))
            })
            .map(|(src, w, tar)| (src, EdgeWeight::Spawn(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges_filter = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(sr, spec.filter(sr)))
            .map(|(src, w, tar)| (src, EdgeWeight::Filter(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges_close = spec
            .outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(sr, spec.close(sr)))
            .map(|(src, w, tar)| (src, EdgeWeight::Close(Box::new(Self::stream_access_kind_to_edge_weight(w))), tar));
        let edges = edges_expr
            .chain(edges_spawn)
            .chain(edges_filter)
            .chain(edges_close)
            .collect::<Vec<(SRef, EdgeWeight, SRef)>>();
        // Check well-formedness = no closed-walk with total weight of zero or positive
        Self::check_well_formedness(&graph)?;
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
        let _dg = DependencyGraph { accesses, accessed_by, aggregates, aggregated_by, graph };
        todo!("Return Dependencies");
    }

    fn check_well_formedness(
        graph: &Graph<SRef, EdgeWeight>,
    ) -> Result<()> {
        let graph = graph_without_negative_offset_edges(graph);
        let graph = graph_without_close_edges(&graph);
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
            ExpressionKind::Window(_) => todo!(),
        }
    }

    fn stream_access_kind_to_edge_weight(w: StreamAccessKind) -> EdgeWeight {
        match w {
            StreamAccessKind::Sync => EdgeWeight::Offset(0),
            StreamAccessKind::Offset(o) => match o {
                Offset::PastDiscreteOffset(o) => EdgeWeight::Offset(-i32::try_from(o).unwrap()),
                Offset::FutureDiscreteOffset(o) => {
                    if o == 0 {
                        EdgeWeight::Offset(i32::try_from(o).unwrap())
                    } else {
                        todo!("implement DG analysis for positive Offsets")
                    }
                }
                _ => todo!(),
            },
            StreamAccessKind::Hold => EdgeWeight::Hold,
            StreamAccessKind::SlidingWindow(wref) => EdgeWeight::Aggr(wref),
            StreamAccessKind::DiscreteWindow(wref) => EdgeWeight::Aggr(wref),
        }
    }
}

#[cfg(test)]
mod tests {
    fn chek_graph_for_spec(_spec: &str, _res: bool) {
        unimplemented!()
    }

    #[test]
    #[ignore]
    fn self_ref() {
        chek_graph_for_spec("input a: Int8\noutput b: Int8 := a+b", false)
    }
    #[test]
    #[ignore]
    fn simple_cycle() {
        chek_graph_for_spec("input a: Int8\noutput b: Int8 := a+d\noutput c: Int8 := b\noutput d: Int8 := c", false)
    }

    #[test]
    #[ignore]
    fn linear_should_be_no_problem() {
        chek_graph_for_spec("input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c", true)
    }

    #[test]
    #[ignore]
    fn negative_cycle_should_be_no_problem() {
        chek_graph_for_spec("output a: Int8 := a.offset(by: -1).defaults(to: 0)", true)
    }

    #[test]
    #[ignore]
    fn self_sliding_window_should_be_no_problem() {
        chek_graph_for_spec("output a: Int8 := a.aggregate(over: 1s, using: sum)", true)
    }

    #[test]
    #[ignore] // Graph Analysis cannot handle positive edges; not required for this branch.
    fn positive_cycle_should_cause_a_warning() {
        chek_graph_for_spec("output a: Int8 := a[1]", false);
    }

    #[test]
    #[ignore]
    fn self_loop() {
        chek_graph_for_spec(
            "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c\noutput e: Int8 := e",
            false,
        )
    }

    #[test]
    #[ignore]
    fn parallel_edges_in_a_cycle() {
        chek_graph_for_spec("input a: Int8\noutput b: Int8 := a+d+d\noutput c: Int8 := b\noutput d: Int8 := c", false)
    }

    #[test]
    #[ignore]
    fn spawn_self_ref() {
        chek_graph_for_spec("input a: Int8\noutput b {spawn if b > 6} := a + 5", false)
    }
    #[test]
    #[ignore]
    fn filter_self_ref() {
        chek_graph_for_spec("input a: Int8\noutput b {filter b > 6} := a + 5", true)
    }

    #[test]
    #[ignore]
    fn close_self_ref() {
        chek_graph_for_spec("input a: Int8\noutput b {close b > 6} := a + 5", true)
    }
}
