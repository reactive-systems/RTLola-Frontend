use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    convert::{TryFrom, TryInto},
    ops::{Add, Mul, Sub},
};

use bitset::BitSet;
use num::ToPrimitive;
use num::{traits::Inv, FromPrimitive};
use ordered_float::NotNan;
use petgraph::{algo::toposort, graph::NodeIndex, prelude::StableGraph, Direction};
use rtlola_parser::ast::Tag;
use rtlola_reporting::{RtLolaError, Span};
use rust_decimal::Decimal;
use uom::si::time::second;
use uom::si::{rational64::Time as UOM_Time, time::nanosecond};

use crate::{
    benchmark::BENCHMARK_TRACER,
    hir::{
        AnnotatedPacingType, ArithLogOp, ConcretePacingType, Constant, DepAnaMode, DepAnaTrait,
        DependencyGraph, EdgeWeight, Eval, ExprId, Expression, ExpressionKind, FnExprKind, Hir,
        Inlined, Literal, Output, OutputKind, SRef, StreamAccessKind, StreamReference, TypedTrait,
    },
    stdlib, BaseMode,
};

fn iterate_cutpoints(
    graph: &AugmentedDependencyGraph,
    cut_set: &HashSet<NodeIndex>,
    visited: &mut HashSet<Vec<NodeIndex>>,
) {
    // canonical key: sorted vector
    let mut key: Vec<NodeIndex> = cut_set.iter().copied().collect();
    key.sort_unstable();
    if !visited.insert(key) {
        return; // already visited
    }

    // snapshot current nodes to iterate
    let elems: Vec<NodeIndex> = cut_set.iter().copied().collect();

    'outer: for &node in &elems {
        // start neighbors from outgoing edges
        let mut neighbors: Vec<NodeIndex> =
            graph.neighbors_directed(node, Direction::Outgoing).collect();

        // check each other node in cut set
        for &n in &elems {
            if n == node {
                continue;
            }

            let n_sr = graph.node_weight(n).unwrap().sref;

            let all_reachable = neighbors.iter().all(|&m| {
                graph.node_weight(m).unwrap().reachable_from.contains(&n_sr)
            });

            let none_reachable = neighbors.iter().all(|&m| {
                !graph.node_weight(m).unwrap().reachable_from.contains(&n_sr)
            });

            if all_reachable {
                // do nothing
            } else if none_reachable {
                neighbors.push(n);
            } else {
                continue 'outer; // inconsistent, skip this node
            }
        }

        // ---- make a fresh copy for recursion ----
        let mut new_cut_set = cut_set.clone();
        new_cut_set.remove(&node);
        new_cut_set.extend(neighbors.iter().copied());

        // recurse
        iterate_cutpoints(graph, &new_cut_set, visited);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyHeuristic {
    Inputs,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
enum SensitivityBound {
    Bounded(NotNan<f64>),
    Unbounded,
}

impl SensitivityBound {
    fn unwrap(self) -> f64 {
        match self {
            SensitivityBound::Bounded(f) => f.into_inner(),
            SensitivityBound::Unbounded => panic!(),
        }
    }
}

impl std::fmt::Display for SensitivityBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Add for SensitivityBound {
    type Output = SensitivityBound;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SensitivityBound::Bounded(lhs), SensitivityBound::Bounded(rhs)) => {
                SensitivityBound::Bounded(lhs + rhs)
            }
            _ => SensitivityBound::Unbounded,
        }
    }
}

impl Mul for SensitivityBound {
    type Output = SensitivityBound;

    fn mul(self, rhs: SensitivityBound) -> Self::Output {
        match (self, rhs) {
            (SensitivityBound::Bounded(b1), SensitivityBound::Bounded(b2)) => {
                SensitivityBound::Bounded(b1 * b2)
            }
            _ => SensitivityBound::Unbounded,
        }
    }
}

impl Mul<u64> for SensitivityBound {
    type Output = SensitivityBound;

    fn mul(self, rhs: u64) -> Self::Output {
        match (self, rhs) {
            (SensitivityBound::Bounded(b1), rhs) => {
                SensitivityBound::Bounded(b1 * NotNan::<f64>::try_from(rhs as f64).unwrap())
            }
            _ => SensitivityBound::Unbounded,
        }
    }
}

impl Default for SensitivityBound {
    fn default() -> Self {
        Self::Bounded(0.0f64.try_into().unwrap())
    }
}

impl From<f64> for SensitivityBound {
    fn from(value: f64) -> Self {
        Self::Bounded(value.try_into().unwrap())
    }
}

impl From<NumInfluencedValues> for SensitivityBound {
    fn from(value: NumInfluencedValues) -> Self {
        match value {
            NumInfluencedValues::Bounded(b) => SensitivityBound::Bounded(b.into()),
            NumInfluencedValues::Unbounded => SensitivityBound::Unbounded,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValueRange {
    Bounded {
        lower: NotNan<f64>,
        upper: NotNan<f64>,
    },
    Unbounded,
}

impl std::fmt::Display for ValueRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ValueRange {
    fn sensitivity(&self) -> SensitivityBound {
        match self {
            ValueRange::Bounded { lower, upper } => {
                SensitivityBound::Bounded((upper - lower).try_into().unwrap())
            }
            ValueRange::Unbounded => SensitivityBound::Unbounded,
        }
    }

    fn constant(f: f64) -> Self {
        ValueRange::Bounded {
            lower: f.try_into().unwrap(),
            upper: f.try_into().unwrap(),
        }
    }
}

impl Add for ValueRange {
    type Output = ValueRange;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                ValueRange::Bounded {
                    lower: l1,
                    upper: u1,
                },
                ValueRange::Bounded {
                    lower: l2,
                    upper: u2,
                },
            ) => ValueRange::Bounded {
                lower: l1 + l2,
                upper: u1 + u2,
            },
            _ => ValueRange::Unbounded,
        }
    }
}

impl Sub for ValueRange {
    type Output = ValueRange;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                ValueRange::Bounded {
                    lower: l1,
                    upper: u1,
                },
                ValueRange::Bounded {
                    lower: l2,
                    upper: u2,
                },
            ) => ValueRange::Bounded {
                lower: l1 - u2,
                upper: u1 - l2,
            },
            _ => ValueRange::Unbounded,
        }
    }
}

impl Mul for ValueRange {
    type Output = ValueRange;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                ValueRange::Bounded {
                    lower: l1,
                    upper: u1,
                },
                ValueRange::Bounded {
                    lower: l2,
                    upper: u2,
                },
            ) => ValueRange::Bounded {
                lower: vec![l1 * l2, l1 * u2, u1 * l2, u1 * u2]
                    .into_iter()
                    .map(|n| NotNan::try_from(n).unwrap())
                    .min()
                    .unwrap(),
                upper: vec![l1 * l2, l1 * u2, u1 * l2, u1 * u2]
                    .into_iter()
                    .map(|n| NotNan::try_from(n).unwrap())
                    .max()
                    .unwrap(),
            },
            _ => ValueRange::Unbounded,
        }
    }
}

impl ValueRange {
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (
                ValueRange::Bounded {
                    lower: l1,
                    upper: u1,
                },
                ValueRange::Bounded {
                    lower: l2,
                    upper: u2,
                },
            ) => ValueRange::Bounded {
                lower: l1.min(l2),
                upper: u1.max(u2),
            },
            _ => ValueRange::Unbounded,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NumInfluencedValues {
    Bounded(u32),
    Unbounded,
}

impl Add for NumInfluencedValues {
    type Output = NumInfluencedValues;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (NumInfluencedValues::Bounded(b1), NumInfluencedValues::Bounded(b2)) => {
                NumInfluencedValues::Bounded(b1 + b2)
            }
            _ => NumInfluencedValues::Unbounded,
        }
    }
}

#[derive(Debug, Clone)]
struct AugmentedNode {
    sref: SRef,
    sensitivity: SensitivityBound,
    value_range: ValueRange,
    reachable_from: HashSet<SRef>
}

type AugmentedDependencyGraph = StableGraph<AugmentedNode, EdgeWeight>;

impl Hir<DepAnaMode> {
    pub(crate) fn add_privacy_barriers(
        mut self,
        parameter: f64,
        heuristic: PrivacyHeuristic,
    ) -> Result<Hir<BaseMode>, RtLolaError> {
        let (loop_free_graph, public_nodes) = if heuristic == PrivacyHeuristic::Inputs {
            let loop_free_graph = self.graph().filter_map(
                |_, n| match n {
                    StreamReference::In(_) => Some(*n),
                    _ => None,
                },
                |_, e| Some(*e),
            );
            let public_nodes = loop_free_graph.node_indices().collect();
            (loop_free_graph, public_nodes)
        } else {
            let loop_free_graph = self.extract_loop_free_segment(self.graph().clone());
            let public_nodes = self.find_public_nodes(&loop_free_graph);
            if public_nodes.is_empty() {
                panic!("no stream marked as public");
            }
            (loop_free_graph, public_nodes)
        };

        let annotated_graph = self.analyze_dependency_graph(loop_free_graph);

        // debug annotations
        for node in annotated_graph.node_indices() {
            let weight = annotated_graph.node_weight(node).unwrap();
            self.add_tag(
                weight.sref,
                "derived_sensitivity",
                Some(&weight.sensitivity.to_string()),
            );
            self.add_tag(
                weight.sref,
                "derived_range",
                Some(&weight.value_range.to_string()),
            );
            if public_nodes.contains(&node) {
                self.add_tag(weight.sref, "derived_public", None);
            }
        }

        let mut inputs: HashSet<_> = annotated_graph
                .node_indices()
                .filter(|i| annotated_graph.node_weight(*i).unwrap().sref.is_input())
                .collect();
            let mut visited =  HashSet::new();
        iterate_cutpoints(&annotated_graph, &mut inputs, &mut visited);

        let mut tracer = BENCHMARK_TRACER.lock().unwrap();
        tracer.start_privacy_heuristic();
        let cut_points = match heuristic {
            PrivacyHeuristic::Inputs => annotated_graph
                .node_indices()
                .filter(|i| annotated_graph.node_weight(*i).unwrap().sref.is_input())
                .collect(),
            PrivacyHeuristic::Deep => public_nodes,
        };
        tracer.end_privacy_heuristic();

        for node in &cut_points {
            let weight = annotated_graph.node_weight(*node).unwrap();
            self.add_noise(
                weight.sref,
                (cut_points.len() as f64 * weight.sensitivity.unwrap()) / parameter,
            );
        }

        let Hir {
            inputs,
            outputs,
            next_input_ref,
            next_output_ref,
            expr_maps,
            global_tags,
            mode: _,
        } = self;

        Ok(Hir {
            inputs,
            outputs,
            next_input_ref,
            next_output_ref,
            expr_maps,
            global_tags,
            mode: BaseMode {},
        })
    }

    fn find_public_nodes(&self, loop_free_graph: &DependencyGraph) -> HashSet<NodeIndex> {
        let loop_free_nodes: HashSet<_> = loop_free_graph.node_indices().collect();

        let public_nodes: HashSet<_> = self
            .graph()
            .node_indices()
            .filter(|p| {
                let sr = self.graph().node_weight(*p).unwrap();
                match sr {
                    StreamReference::In(_) => false,
                    StreamReference::Out(o) => {
                        matches!(self.outputs[*o].kind, OutputKind::Trigger(_))
                            || self.outputs[*o].tags.contains_key("public")
                    }
                }
            })
            .collect();

        let mut public_loop_free_nodes = HashSet::new();
        for node in public_nodes {
            let mut visited_nodes = HashSet::new();
            Self::find_public_nodes_prime(
                self.graph(),
                node,
                &loop_free_nodes,
                &mut public_loop_free_nodes,
                &mut visited_nodes,
            );
        }

        public_loop_free_nodes
    }

    fn find_public_nodes_prime(
        graph: &DependencyGraph,
        cur_node: NodeIndex,
        loop_free_nodes: &HashSet<NodeIndex>,
        public_loop_free_nodes: &mut HashSet<NodeIndex>,
        visited_nodes: &mut HashSet<NodeIndex>,
    ) {
        if loop_free_nodes.contains(&cur_node) {
            public_loop_free_nodes.insert(cur_node);
            return;
        }
        if visited_nodes.contains(&cur_node) {
            return;
        }
        visited_nodes.insert(cur_node);

        for node in graph.neighbors_directed(cur_node, Direction::Outgoing) {
            Self::find_public_nodes_prime(
                graph,
                node,
                loop_free_nodes,
                public_loop_free_nodes,
                visited_nodes,
            );
        }
    }

    fn extract_loop_free_segment(&mut self, mut graph: DependencyGraph) -> DependencyGraph {
        let cycle_nodes = self.find_cycle_nodes(&graph);
        let all_aggregations = self.find_all_aggregation_sucessors(&graph);

        let mut queue = cycle_nodes
            .into_iter()
            .chain(all_aggregations)
            .collect::<VecDeque<NodeIndex>>();
        let mut visited_nodes = HashSet::new();
        while let Some(idx) = queue.pop_front() {
            if visited_nodes.contains(&idx) {
                continue;
            }
            visited_nodes.insert(idx);
            for neighbor in graph.neighbors_directed(idx, Direction::Incoming) {
                queue.push_back(neighbor);
            }
        }

        for node in visited_nodes {
            graph.remove_node(node);
        }
        graph
    }

    fn find_all_aggregation_sucessors(&mut self, graph: &DependencyGraph) -> Vec<NodeIndex> {
        graph
            .node_indices()
            .filter(|i| {
                let weight = graph.node_weight(*i).unwrap();
                let is_all_aggregation = match *weight {
                    StreamReference::In(_) => {
                        return false;
                    }
                    StreamReference::Out(_) => self.eval_expr(*weight).unwrap().iter().any(|e| {
                        matches!(
                            &e.kind,
                            ExpressionKind::StreamAccess(_, StreamAccessKind::AllAggregation(_), _)
                        )
                    }),
                };
                is_all_aggregation
            })
            .flat_map(|a| graph.neighbors_directed(a, Direction::Incoming))
            .collect()
    }

    fn find_cycle_nodes(&mut self, graph: &DependencyGraph) -> Vec<NodeIndex> {
        let mut visited = HashSet::new();
        let mut cycle_nodes = HashSet::new();

        for node in graph.node_indices() {
            let mut path = Vec::new();
            Self::find_cycle_nodes_prime(graph, node, &mut path, &mut visited, &mut cycle_nodes);
        }

        for node in &cycle_nodes {
            let sr = graph.node_weight(*node).unwrap();
            self.add_tag(*sr, "is_cycle", None);
        }
        cycle_nodes.into_iter().collect()
    }

    fn find_cycle_nodes_prime(
        graph: &DependencyGraph,
        cur_node: NodeIndex,
        path: &mut Vec<NodeIndex>,
        visited: &mut HashSet<NodeIndex>,
        cycle_nodes: &mut HashSet<NodeIndex>,
    ) {
        if let Some(position) = path.iter().position(|n| n == &cur_node) {
            cycle_nodes.extend(&path[position..]);
        }
        if visited.contains(&cur_node) {
            return;
        }
        visited.insert(cur_node);
        path.push(cur_node);

        for neighbor in graph.neighbors_directed(cur_node, Direction::Incoming) {
            Self::find_cycle_nodes_prime(graph, neighbor, path, visited, cycle_nodes);
        }

        path.pop();
    }

    fn analyze_dependency_graph(&self, graph: DependencyGraph) -> AugmentedDependencyGraph {
        let (mut sensitivities, mut value_ranges) = self.get_input_annotations();
        let mut num_influenced_values: HashMap<_, _> = self
            .inputs
            .iter()
            .map(|i| (i.sr, NumInfluencedValues::Bounded(1)))
            .collect();
        let mut reachable_from: HashMap<_,HashSet<_>> = self.inputs.iter().map(|i| (i.sr, vec![i.sr].into_iter().collect())).collect();

        for node in toposort(&graph, None).expect("no cycles").into_iter().rev() {
            let sr = *graph.node_weight(node).unwrap();
            if let StreamReference::In(_) = sr {
                // already handled
                continue;
            }
            let eval_clauses = self.eval_expr(sr).unwrap();
            assert_eq!(
                eval_clauses.len(),
                1,
                "multiple eval clauses not supported for privacy analysis"
            );
            let expression = eval_clauses[0];
            let value_range = Self::calculate_value_range(expression, &value_ranges);
            let num_influenced_value =
                Self::calculate_num_influenced_values(expression, &num_influenced_values);
            let sensitivity = self.calculate_sensitivity(
                expression,
                &sensitivities,
                &value_range,
                num_influenced_value,
            );
            value_ranges.insert(sr, value_range);
            sensitivities.insert(sr, sensitivity);
            num_influenced_values.insert(sr, num_influenced_value);

            let reachable_set = graph.neighbors_directed(node, Direction::Outgoing).flat_map(|n| {
                reachable_from[graph.node_weight(n).unwrap()].iter()
            }).copied().chain(Some(sr)).collect();
            reachable_from.insert(sr, reachable_set);
        }

        let graph = graph.map(
            |_, &sref| AugmentedNode {
                sref,
                sensitivity: *sensitivities
                    .get(&sref)
                    .expect("each node should have a sensitivity"),
                value_range: *value_ranges
                    .get(&sref)
                    .expect("each node should have a value range"),
                reachable_from: reachable_from.remove(&sref).unwrap_or_default()
            },
            |_, e| *e,
        );

        graph
    }

    fn calculate_sensitivity(
        &self,
        expression: &Expression,
        sensitivities: &HashMap<SRef, SensitivityBound>,
        value_range: &ValueRange,
        num_influenced_values: NumInfluencedValues,
    ) -> SensitivityBound {
        match &expression.kind {
            ExpressionKind::LoadConstant(_) => 0.0.into(),
            ExpressionKind::ArithLog(ArithLogOp::Add | ArithLogOp::Sub, subexps) => {
                assert_eq!(subexps.len(), 2);
                let lhs = &subexps[0];
                let rhs = &subexps[1];
                let lhs_sens = self.calculate_sensitivity(
                    lhs,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                let rhs_sens = self.calculate_sensitivity(
                    rhs,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                lhs_sens + rhs_sens
            }
            ExpressionKind::StreamAccess(
                sr,
                StreamAccessKind::Sync | StreamAccessKind::Offset(_),
                _,
            ) => *sensitivities
                .get(sr)
                .expect("dependencies should already be processed"),
            ExpressionKind::StreamAccess(_sr, StreamAccessKind::Hold, _) => {
                SensitivityBound::Unbounded
            }
            ExpressionKind::StreamAccess(sr, StreamAccessKind::SlidingWindow(w), _) => {
                let w = self.single_sliding(*w);
                let window_duration = w.aggr.duration.as_nanos() as u64;
                let caller_pacing = self.eval_pacing_type(w.caller, 0);
                let pacing_nanos = match caller_pacing {
                    ConcretePacingType::FixedGlobalPeriodic(freq)
                    | ConcretePacingType::FixedLocalPeriodic(freq) => {
                        UOM_Time::new::<second>(freq.get::<uom::si::frequency::hertz>().inv())
                            .get::<nanosecond>()
                    }
                    _ => unreachable!(),
                }
                .to_integer()
                .to_u64()
                .unwrap();
                let factor = window_duration.div_ceil(pacing_nanos);
                *sensitivities
                    .get(sr)
                    .expect("dependencies should already be processed")
                    * factor
            }
            ExpressionKind::Default { expr, default } => {
                let expr_bound = self.calculate_sensitivity(
                    expr,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                let default_bound = self.calculate_sensitivity(
                    default,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                expr_bound.max(default_bound)
            }
            _ => value_range.sensitivity() * SensitivityBound::from(num_influenced_values),
        }
    }

    fn calculate_value_range(
        expression: &Expression,
        value_ranges: &HashMap<SRef, ValueRange>,
    ) -> ValueRange {
        match &expression.kind {
            ExpressionKind::LoadConstant(
                Constant::Basic(literal) | Constant::Inlined(Inlined { lit: literal, .. }),
            ) => match literal {
                Literal::Bool(true) => ValueRange::constant(0.0),
                Literal::Bool(false) => ValueRange::constant(1.0),
                Literal::Integer(i) => ValueRange::constant((*i) as f64),
                Literal::SInt(i) => ValueRange::constant((*i) as f64),
                Literal::Decimal(decimal) => ValueRange::constant(decimal.to_f64().unwrap()),
                Literal::Str(_) => ValueRange::Unbounded,
            },
            ExpressionKind::ArithLog(
                op @ (ArithLogOp::Add | ArithLogOp::Sub | ArithLogOp::Mul),
                subexps,
            ) => {
                let lhs = &subexps[0];
                let rhs = &subexps[1];
                let lhs_range = Self::calculate_value_range(lhs, value_ranges);
                let rhs_range = Self::calculate_value_range(rhs, value_ranges);
                match op {
                    ArithLogOp::Add => lhs_range + rhs_range,
                    ArithLogOp::Sub => lhs_range - rhs_range,
                    ArithLogOp::Mul => lhs_range * rhs_range,
                    _ => unimplemented!(),
                }
            }
            ExpressionKind::StreamAccess(
                target,
                StreamAccessKind::Sync | StreamAccessKind::Offset(_) | StreamAccessKind::Hold,
                _,
            ) => value_ranges[target],
            ExpressionKind::StreamAccess(
                _,
                StreamAccessKind::SlidingWindow(_) | StreamAccessKind::AllAggregation(_),
                _,
            ) => ValueRange::Unbounded,
            ExpressionKind::Default { expr, default } => {
                Self::calculate_value_range(expr, value_ranges)
                    .union(Self::calculate_value_range(default, value_ranges))
            }
            _ => ValueRange::Unbounded,
        }
    }

    fn calculate_num_influenced_values(
        expression: &Expression,
        num_influenced_values: &HashMap<SRef, NumInfluencedValues>,
    ) -> NumInfluencedValues {
        match &expression.kind {
            ExpressionKind::LoadConstant(_) => NumInfluencedValues::Bounded(0),
            ExpressionKind::ArithLog(_, exprs) => exprs
                .iter()
                .map(|e| Self::calculate_num_influenced_values(e, num_influenced_values))
                .reduce(|a, b| a + b)
                .unwrap(),
            ExpressionKind::StreamAccess(
                target,
                StreamAccessKind::Offset(_) | StreamAccessKind::Sync,
                _,
            ) => num_influenced_values[&target],
            _ => NumInfluencedValues::Unbounded,
        }
    }

    fn add_noise(&mut self, stream: SRef, amount: f64) {
        match stream {
            StreamReference::In(idx) => {
                let input_id = self.next_expr_id();
                let input = &self.inputs[idx];
                let expr = Expression {
                    kind: ExpressionKind::StreamAccess(input.sr, StreamAccessKind::Sync, vec![]),
                    eid: input_id,
                    span: Span::Unknown,
                };
                self.expr_maps.exprid_to_expr.insert(input_id, expr);
                let sr = StreamReference::Out(self.outputs.len());
                let copy = Output {
                    kind: OutputKind::NamedOutput(format!("{}'", input.name)),
                    annotated_type: None,
                    params: vec![],
                    spawn: None,
                    eval: vec![Eval {
                        annotated_pacing_type: AnnotatedPacingType::NotAnnotated(Span::Unknown),
                        condition: None,
                        expr: input_id,
                        span: Span::Unknown,
                    }],
                    close: None,
                    sr,
                    tags: HashMap::new(),
                    span: Span::Unknown,
                };
                self.outputs.push(copy);
                self.replace_sr(stream, sr, input_id);
                self.add_noise(sr, amount);
            }
            StreamReference::Out(idx) => {
                assert!(
                    self.outputs[idx].eval.len() == 1,
                    "for now we only support single eval clauses"
                );

                let old_expr = self.outputs[idx].eval[0].expr;

                let amount_id = self.next_expr_id();
                let amount_expr = Expression {
                    kind: ExpressionKind::LoadConstant(Constant::Basic(Literal::Decimal(
                        Decimal::from_f64(amount).unwrap(),
                    ))),
                    eid: amount_id,
                    span: Span::Unknown,
                };
                self.expr_maps
                    .exprid_to_expr
                    .insert(amount_id, amount_expr.clone());

                let noise_expr_id = self.next_expr_id();
                let noise_expr = Expression {
                    kind: ExpressionKind::Function(FnExprKind {
                        name: "laplace".into(),
                        args: vec![amount_expr],
                        type_param: vec![],
                    }),
                    eid: noise_expr_id,
                    span: Span::Unknown,
                };
                self.expr_maps
                    .exprid_to_expr
                    .insert(noise_expr_id, noise_expr.clone());

                let new_expr_id = self.next_expr_id();
                let new_expr = Expression {
                    kind: ExpressionKind::ArithLog(
                        ArithLogOp::Add,
                        vec![self.expr_maps.exprid_to_expr[&old_expr].clone(), noise_expr],
                    ),
                    eid: new_expr_id,
                    span: Span::Unknown,
                };
                self.expr_maps.exprid_to_expr.insert(new_expr_id, new_expr);
                self.outputs[idx].eval[0].expr = new_expr_id;
                self.expr_maps.func_table.insert(
                    "laplace".into(),
                    (*stdlib::noise_module()
                        .iter()
                        .find(|s| s.name.name() == "laplace")
                        .unwrap())
                    .clone(),
                );
            }
        }
    }

    fn next_expr_id(&mut self) -> ExprId {
        let id = self
            .expr_maps
            .exprid_to_expr
            .keys()
            .max()
            .map(|v| v.0 + 1)
            .unwrap_or(0);
        ExprId(id)
    }

    fn replace_sr(&mut self, from: StreamReference, to: StreamReference, exclude: ExprId) {
        for (id, expr) in self.expr_maps.exprid_to_expr.iter_mut() {
            if *id == exclude {
                continue;
            }
            Self::replace_sr_prime(expr, from, to);
        }
    }

    fn replace_sr_prime(expr: &mut Expression, from: StreamReference, to: StreamReference) {
        match &mut expr.kind {
            ExpressionKind::LoadConstant(_) => {}
            ExpressionKind::Tuple(expressions) | ExpressionKind::ArithLog(_, expressions) => {
                for expr in expressions {
                    Self::replace_sr_prime(expr, from, to);
                }
            }
            ExpressionKind::StreamAccess(stream_reference, _, expressions) => {
                if *stream_reference == from {
                    *stream_reference = to;
                }
                for expr in expressions {
                    Self::replace_sr_prime(expr, from, to);
                }
            }
            ExpressionKind::ParameterAccess(_, _) => {}
            ExpressionKind::LambdaParameterAccess { .. } => {}
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                for expr in [condition, consequence, alternative] {
                    Self::replace_sr_prime(expr, from, to);
                }
            }
            ExpressionKind::Function(fn_expr_kind) => {
                for expr in &mut fn_expr_kind.args {
                    Self::replace_sr_prime(expr, from, to);
                }
            }
            ExpressionKind::Widen(_) => todo!(),
            ExpressionKind::Default { expr, default } => {
                for expr in [expr, default] {
                    Self::replace_sr_prime(expr, from, to);
                }
            }
            ExpressionKind::TupleAccess(expression, _) => {
                Self::replace_sr_prime(expression, from, to);
            }
        }
    }

    fn get_input_annotations(
        &self,
    ) -> (HashMap<SRef, SensitivityBound>, HashMap<SRef, ValueRange>) {
        let mut sensitivities = HashMap::new();
        let mut value_ranges = HashMap::new();

        for input in &self.inputs {
            let value_range_from = input.tags.get("range_from");
            let value_range_to = input.tags.get("range_to");
            let value_range = match (value_range_from, value_range_to) {
                (Some(lower), Some(upper)) => {
                    let lower = lower.value.as_ref().expect("lower bound must have a value");
                    let lower: f64 = lower.parse().expect("lower bound must be a number");
                    let upper = upper.value.as_ref().expect("upper bound must have a value");
                    let upper: f64 = upper.parse().expect("upper bound must be a number");
                    ValueRange::Bounded {
                        lower: lower.try_into().unwrap(),
                        upper: upper.try_into().unwrap(),
                    }
                }
                (None, None) => ValueRange::Unbounded,
                _ => panic!("Both lower and upper must be annotated for value range"),
            };

            value_ranges.insert(input.sr, value_range);

            if let ValueRange::Bounded { .. } = value_range {
                assert!(
                    input.tags.get("sensitivity").is_none(),
                    "give either sensitivity or range on input"
                );
                let sensitivity = value_range.sensitivity();
                sensitivities.insert(input.sr, sensitivity);
            } else {
                let sensitivity = input
                    .tags
                    .get("sensitivity")
                    .expect("each unput must be annotated with a sensitivity");
                let sensitivity = sensitivity
                    .value
                    .as_ref()
                    .expect("sensitivity annotation must have a value");
                let sensitivity: f64 = sensitivity
                    .parse()
                    .expect("sensitivity annotation must be a number");
                sensitivities.insert(input.sr, sensitivity.try_into().unwrap());
            }
        }

        (sensitivities, value_ranges)
    }

    fn add_tag(&mut self, sr: SRef, key: &str, value: Option<&str>) {
        let tags = match sr {
            StreamReference::In(i) => &mut self.inputs[i].tags,
            StreamReference::Out(o) => &mut self.outputs[o].tags,
        };
        tags.insert(
            key.into(),
            Tag {
                key: key.into(),
                value: value.map(|v| v.into()),
                span: Span::Unknown,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::{HashMap, HashSet}, convert::TryInto};

    use rtlola_parser::{ParserConfig, parse};

    use crate::{from_ast, hir::{DepAnaTrait, HirStage, StreamReference}, modes::privacy::{SensitivityBound, ValueRange, iterate_cutpoints}};

    #[test]
    fn simple_spec() {
        let spec = "
        #[range_from=\"0\", range_to=\"5\"]
        input a : UInt64
        output b := a + 1
        output c := a + 2
        output d := b + c";
        let config = ParserConfig::for_string(spec.into());
        let ast = parse(&config).unwrap();
        let base = from_ast(ast).unwrap();
        let config = (&config).into();
        let hir = base.progress(&config).unwrap().progress(&config).unwrap();
        let graph = hir.analyze_dependency_graph(hir.graph().clone());
        let nodes : HashMap<_, _> = graph.node_indices().map(|n| {
            let w = graph.node_weight(n).unwrap();
        (w.sref, w)}).collect();
        assert_eq!(nodes[&StreamReference::In(0)].reachable_from, vec![StreamReference::In(0)].into_iter().collect());
        assert_eq!(nodes[&StreamReference::In(0)].sensitivity, SensitivityBound::from(5.0));
        assert_eq!(nodes[&StreamReference::In(0)].value_range, ValueRange::Bounded { lower: 0.0f64.try_into().unwrap(), upper: 5.0f64.try_into().unwrap() });

        assert_eq!(nodes[&StreamReference::Out(0)].reachable_from, vec![StreamReference::In(0), StreamReference::Out(0)].into_iter().collect());
        assert_eq!(nodes[&StreamReference::Out(0)].sensitivity, SensitivityBound::from(5.0));
        assert_eq!(nodes[&StreamReference::Out(0)].value_range, ValueRange::Bounded { lower: 1.0f64.try_into().unwrap(), upper: 6.0f64.try_into().unwrap() });

        assert_eq!(nodes[&StreamReference::Out(1)].reachable_from, vec![StreamReference::In(0), StreamReference::Out(1)].into_iter().collect());
        assert_eq!(nodes[&StreamReference::Out(1)].sensitivity, SensitivityBound::from(5.0));
        assert_eq!(nodes[&StreamReference::Out(1)].value_range, ValueRange::Bounded { lower: 2.0f64.try_into().unwrap(), upper: 7.0f64.try_into().unwrap() });

        assert_eq!(nodes[&StreamReference::Out(2)].reachable_from, vec![StreamReference::In(0), StreamReference::Out(0), StreamReference::Out(1), StreamReference::Out(2)].into_iter().collect());
        assert_eq!(nodes[&StreamReference::Out(2)].sensitivity, SensitivityBound::from(10.0));
        assert_eq!(nodes[&StreamReference::Out(2)].value_range, ValueRange::Bounded { lower: 3.0f64.try_into().unwrap(), upper: 13.0f64.try_into().unwrap() });

        let mut inputs: HashSet<_> = graph
                .node_indices()
                .filter(|i| graph.node_weight(*i).unwrap().sref.is_input())
                .collect();
        iterate_cutpoints(&graph, &mut inputs, &mut HashSet::new());
    }
}
