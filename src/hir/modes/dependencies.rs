use crate::common_ir::{Layer, SRef, WRef};

use super::Dependencies;

use std::collections::HashMap;

use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::{expression::StreamAccessKind, Hir};
use crate::{
    common_ir::Offset,
    hir::modes::{ir_expr::WithIrExpr, HirMode},
};
use petgraph::{
    algo::{bellman_ford, FloatMeasure, NegativeCycle},
    graph::NodeIndex,
    Graph,
};

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
    fn accesses(&self, _who: SRef) -> Vec<SRef> {
        todo!()
    }

    fn accessed_by(&self, _who: SRef) -> Vec<SRef> {
        todo!()
    }

    fn aggregated_by(&self, _who: SRef) -> Vec<(SRef, WRef)> {
        todo!()
    }

    fn aggregates(&self, _who: SRef) -> Vec<(SRef, WRef)> {
        todo!()
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
}

pub(crate) struct DependencyReport {}

type Result<T> = std::result::Result<T, DependencyErr>;
type DG = Graph<SRef, EdgeWeight>;

pub(crate) struct DependencyGraph {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeWeight {
    Infinite,
    Finite(u32),
}

impl FloatMeasure for EdgeWeight {
    fn zero() -> Self {
        EdgeWeight::Finite(0)
    }

    fn infinite() -> Self {
        EdgeWeight::Infinite
    }
}

impl std::ops::Add for EdgeWeight {
    type Output = EdgeWeight;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (EdgeWeight::Infinite, _) | (_, EdgeWeight::Infinite) => EdgeWeight::Infinite,
            (EdgeWeight::Finite(w1), EdgeWeight::Finite(w2)) => EdgeWeight::Finite(w1 + w2),
        }
    }
}

impl PartialOrd for EdgeWeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (EdgeWeight::Infinite, EdgeWeight::Infinite) => None,
            (EdgeWeight::Infinite, EdgeWeight::Finite(_)) => Some(std::cmp::Ordering::Greater),
            (EdgeWeight::Finite(_), EdgeWeight::Infinite) => Some(std::cmp::Ordering::Less),
            (EdgeWeight::Finite(w1), EdgeWeight::Finite(w2)) => w1.partial_cmp(w2),
        }
    }
}

impl Default for EdgeWeight {
    fn default() -> Self {
        EdgeWeight::Finite(0)
    }
}

impl Dependencies {
    pub(crate) fn analyze<M: HirMode + 'static + WithIrExpr>(spec: &Hir<M>) -> Result<Dependencies>
    where
        M: WithIrExpr + HirMode,
    {
        let num_nodes = spec.num_inputs() + spec.num_outputs() + spec.num_triggers();
        let num_edges = num_nodes; // Todo: improve estimate.
        let mut graph: DG = Graph::with_capacity(num_nodes, num_edges);
        let mapping: HashMap<SRef, NodeIndex> = spec.all_streams().map(|sr| (sr, graph.add_node(sr))).collect();
        spec.outputs()
            .map(|o| o.sr)
            .chain(spec.triggers().map(|t| t.sr))
            .flat_map(|sr| Self::collect_edges(sr, spec.expr(sr)))
            .map(|(src, w, tar)| (mapping[&src], w, mapping[&tar]))
            .for_each(|(src, w, tar)| {
                let _ = graph.add_edge(src, tar, EdgeWeight::Finite(w));
            });

        Self::check_negative_cycle(&graph, &spec, &mapping)?;
        todo!()
    }

    fn check_negative_cycle<M: HirMode + WithIrExpr>(
        graph: &DG,
        spec: &Hir<M>,
        mapping: &HashMap<SRef, NodeIndex>,
    ) -> Result<()> {
        match spec
        .outputs()
        .map(|o| bellman_ford(&graph, mapping[&o.sr]))
        .collect::<std::result::Result<Vec<(Vec<EdgeWeight>, Vec<Option<NodeIndex>>)>, NegativeCycle>>() // In essence: err = negative cycle, ok = no negative cycle.
    {
        Err(_) => Err(DependencyErr::NegativeCycle),
        Ok(_) => todo!(),
    }
    }

    fn collect_edges(src: SRef, expr: &Expression) -> Vec<(SRef, u32, SRef)> {
        match &expr.kind {
            ExpressionKind::StreamAccess(target, StreamAccessKind::Sync)
            | ExpressionKind::StreamAccess(target, StreamAccessKind::DiscreteWindow(_)) => vec![(src, 0, *target)],
            ExpressionKind::StreamAccess(_target, StreamAccessKind::Hold)
            | ExpressionKind::StreamAccess(_target, StreamAccessKind::SlidingWindow(_)) => Vec::new(),
            ExpressionKind::StreamAccess(target, StreamAccessKind::Offset(offset)) => {
                vec![(src, Self::offset_to_weight(offset), *target)]
            }
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
