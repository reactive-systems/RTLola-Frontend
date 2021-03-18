#![allow(dead_code)]

use std::collections::HashMap;

use crate::common_ir::StreamAccessKind;
use crate::hir::Hir;
use crate::{
    common_ir::Offset,
    hir::modes::{IrExprTrait, HirMode},
};
use crate::{
    common_ir::SRef,
    hir::expression::{Expression, ExpressionKind},
};
use petgraph::{
    algo::{bellman_ford, FloatMeasure, NegativeCycle},
    graph::NodeIndex,
    Graph,
};

pub(crate) enum DependencyError {
    NegativeCycle, // Should probably contain the cycle
}

pub(crate) struct DependencyReport {}

type Result<T> = std::result::Result<T, DependencyError>;
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

impl DependencyGraph {
    pub(crate) fn analyze<M: HirMode + 'static + IrExprTrait>(spec: &Hir<M>) -> Result<DependencyReport>
    where
        M: IrExprTrait + HirMode,
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

    fn check_negative_cycle<M: HirMode + IrExprTrait>(
        graph: &DG,
        spec: &Hir<M>,
        mapping: &HashMap<SRef, NodeIndex>,
    ) -> Result<()> {
        match spec
        .outputs()
        .map(|o| bellman_ford(&graph, mapping[&o.sr]))
        .collect::<std::result::Result<Vec<(Vec<EdgeWeight>, Vec<Option<NodeIndex>>)>, NegativeCycle>>() // In essence: err = negative cycle, ok = no negative cycle.
    {
        Err(_) => Err(DependencyError::NegativeCycle),
        Ok(_) => todo!(),
    }
    }

    fn collect_edges(src: SRef, expr: &Expression) -> Vec<(SRef, u32, SRef)> {
        match &expr.kind {
            ExpressionKind::StreamAccess(target, StreamAccessKind::Sync, _)
            | ExpressionKind::StreamAccess(target, StreamAccessKind::DiscreteWindow(_), _) => vec![(src, 0, *target)],
            ExpressionKind::StreamAccess(_target, StreamAccessKind::Hold, _)
            | ExpressionKind::StreamAccess(_target, StreamAccessKind::SlidingWindow(_), _) => Vec::new(),
            ExpressionKind::StreamAccess(target, StreamAccessKind::Offset(offset), _) => {
                vec![(src, Self::offset_to_weight(offset), *target)]
            }
            ExpressionKind::LoadConstant(_) => Vec::new(),
            ExpressionKind::ArithLog(_op, args) => {
                args.iter().flat_map(|a| Self::collect_edges(src, a).into_iter()).collect()
            }
            ExpressionKind::Tuple(content) => content.iter().flat_map(|a| Self::collect_edges(src, a)).collect(),
            ExpressionKind::Function { name: _, args, type_param: _ } => {
                args.iter().flat_map(|a| Self::collect_edges(src, a)).collect()
            }
            ExpressionKind::Ite { condition, consequence, alternative } => Self::collect_edges(src, condition)
                .into_iter()
                .chain(Self::collect_edges(src, consequence).into_iter())
                .chain(Self::collect_edges(src, alternative).into_iter())
                .collect(),
            ExpressionKind::TupleAccess(content, _n) => Self::collect_edges(src, content),
            ExpressionKind::Widen(inner, _) => Self::collect_edges(src, inner),
            ExpressionKind::Default { expr, default } => Self::collect_edges(src, expr)
                .into_iter()
                .chain(Self::collect_edges(src, default).into_iter())
                .collect(),
            ExpressionKind::ParameterAccess(_, _) => todo!(),
        }
    }

    fn offset_to_weight(_o: &Offset) -> u32 {
        todo!()
    }
}
