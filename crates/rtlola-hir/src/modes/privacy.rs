use std::{
    collections::{HashMap, HashSet, VecDeque},
    convert::{TryFrom, TryInto},
    ops::{Add, Mul, Sub},
};

use num::ToPrimitive;
use num::{traits::Inv, FromPrimitive};
use ordered_float::NotNan;
use petgraph::{algo::toposort, graph::NodeIndex, prelude::StableGraph, Direction};
use rtlola_parser::ast::{Tag, WindowOperation};
use rtlola_reporting::{Diagnostic, RtLolaError, Span};
use rust_decimal::Decimal;
use uom::si::time::second;
use uom::si::{rational64::Time as UOM_Time, time::nanosecond};

use crate::{
    benchmark::BENCHMARK_TRACER,
    hir::{
        AnnotatedPacingType, ArithLogOp, ConcretePacingType, Constant, DepAnaMode, DepAnaTrait,
        DependencyGraph, EdgeWeight, Eval, ExprId, Expression, ExpressionKind, FnExprKind, Hir,
        Inlined, InputReference, Literal, Origin, Output, OutputKind, SRef, StreamAccessKind,
        StreamReference, TypedTrait,
    },
    stdlib, BaseMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The heuristic that is used to decide where barriers are placed in the specification
pub enum PrivacyHeuristic {
    /// Add barriers directly after the inputs
    Inputs,
    /// Add barriers as close as possible to the outputs
    Deep,
    /// Add the least number of barriers possible
    LeastCutpoints,
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

    fn is_bounded(&self) -> bool {
        match self {
            SensitivityBound::Bounded(_) => true,
            SensitivityBound::Unbounded => false,
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
            ValueRange::Bounded { lower, upper } => SensitivityBound::Bounded(upper - lower),
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
                    .min()
                    .unwrap(),
                upper: vec![l1 * l2, l1 * u2, u1 * l2, u1 * u2]
                    .into_iter()
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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
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
}

type AugmentedDependencyGraph = StableGraph<AugmentedNode, EdgeWeight>;

impl Hir<DepAnaMode> {
    pub(crate) fn add_privacy_barriers(
        mut self,
        parameter: f64,
        heuristic: PrivacyHeuristic,
    ) -> Result<Hir<BaseMode>, RtLolaError> {
        // Fast path for Inputs heuristic
        if heuristic == PrivacyHeuristic::Inputs {
            let (sensitivities, _) = self.get_input_annotations()?;
            if let Some((sr, _)) = sensitivities
                .iter()
                .find(|&(_, s)| *s == SensitivityBound::Unbounded)
            {
                let i = self.input(*sr).unwrap();
                return Err(Diagnostic::error("For input sensitivity each input has to be annotated with a bounded sensitivity").add_span_with_label(i.span(), Some(&format!("The input \"{}\" has unbounded sensitivity.", i.name)), true).into());
            }
            let input_refs = self.inputs.iter().map(|i| i.sr).collect::<Vec<_>>();
            let input_noise = input_refs
                .into_iter()
                .map(|i| {
                    (
                        i.in_ix(),
                        self.inputs.len() as f64 * sensitivities[&i].unwrap() / parameter,
                    )
                })
                .collect::<Vec<_>>();
            self.add_noise_to_inputs(&input_noise);

            return Ok(Hir {
                inputs: self.inputs,
                outputs: self.outputs,
                next_input_ref: self.next_input_ref,
                next_output_ref: self.next_output_ref,
                expr_maps: self.expr_maps,
                global_tags: self.global_tags,
                mode: BaseMode {},
            });
        }

        let loop_free_graph = self.extract_loop_free_segment(self.graph().clone());
        let public_nodes = self.find_public_nodes(&loop_free_graph);
        if public_nodes.is_empty() {
            return Err(Diagnostic::error("At least one output stream has to be marked with #[public] in order to use privacy features.").into());
        }

        let annotated_graph = self.analyze_dependency_graph(loop_free_graph)?;
        let annotated_graph = annotated_graph.filter_map(
            |_, n| Some(n.clone()),
            |_, e| (e.origin != Origin::Spawn).then_some(*e),
        );

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

        if true {
            let mut tracer = BENCHMARK_TRACER.lock().unwrap();
            tracer.start_privacy_heuristic();

            let no_cutpoint_error = Diagnostic::error("No valid set of barrier found. This is possible if some inputs are annotated with \"unbounded\".").into();

            let cut_points = match heuristic {
                PrivacyHeuristic::Inputs => unreachable!("fast path above"),
                PrivacyHeuristic::Deep => {
                    let cutpoints = Self::enumerate_cuts(&annotated_graph, true);
                    if cutpoints.len() < 1 {
                        return Err(no_cutpoint_error);
                    }
                    assert_eq!(cutpoints.len(), 1);
                    cutpoints.into_iter().next().unwrap()
                }
                PrivacyHeuristic::LeastCutpoints => {
                    let cutpoints = Self::enumerate_cuts(&annotated_graph, false);
                    cutpoints
                        .into_iter()
                        .min_by_key(|c| c.len())
                        .ok_or(no_cutpoint_error)?
                }
            };
            tracer.end_privacy_heuristic();

            for node in &cut_points {
                let weight = annotated_graph.node_weight(*node).unwrap();
                self.add_noise(
                    weight.sref,
                    (cut_points.len() as f64 * weight.sensitivity.unwrap()) / parameter,
                );
            }
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

    fn enumerate_cuts(
        graph: &AugmentedDependencyGraph,
        return_first: bool,
    ) -> Vec<HashSet<NodeIndex>> {
        let mut cuts = Vec::new();

        // toposort: outputs first, inputs last
        let topo = toposort(graph, None).unwrap();

        let mut remaining_succ: HashMap<_, _> = graph
            .node_indices()
            .map(|n| (n, graph.neighbors_directed(n, Direction::Outgoing).count()))
            .collect();

        let mut frontier: HashSet<_> = graph
            .node_indices()
            .filter(|n| graph.neighbors_directed(*n, Direction::Incoming).count() == 0)
            .collect();

        let mut d: HashSet<_> = HashSet::new();

        for node in topo {
            frontier.insert(node);

            for downstream in graph.neighbors_directed(node, Direction::Incoming) {
                d.insert(downstream);
                let count = remaining_succ.get_mut(&downstream).unwrap();
                *count -= 1;
                if *count == 0 {
                    frontier.remove(&downstream);
                }
            }

            if !frontier.iter().any(|n| d.contains(n))
                && frontier
                    .iter()
                    .all(|n| graph.node_weight(*n).unwrap().sensitivity.is_bounded())
            {
                cuts.push(frontier.clone());
                if return_first {
                    break;
                }
            }
        }

        cuts
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

    fn analyze_dependency_graph(
        &self,
        graph: DependencyGraph,
    ) -> Result<AugmentedDependencyGraph, RtLolaError> {
        let (mut sensitivities, mut value_ranges) = self.get_input_annotations()?;
        let mut num_influenced_values: HashMap<_, _> = self
            .inputs
            .iter()
            .map(|i| (i.sr, NumInfluencedValues::Bounded(1)))
            .collect();

        let toposort = toposort(&graph, None).expect("no cycles");
        for &node in toposort.iter().rev() {
            let sr = *graph.node_weight(node).unwrap();
            if let StreamReference::In(_) = sr {
                // already handled
                continue;
            }
            let eval_clauses = self.eval_expr(sr).unwrap();
            if eval_clauses.len() > 1 {
                return Err(Diagnostic::error(
                    "Multiple Eval clauses are not supported for privacy analysis.",
                )
                .add_span_with_label(
                    eval_clauses[1].span(),
                    Some("Found second eval clause here".into()),
                    true,
                )
                .into());
            }
            let expression = eval_clauses[0];
            let value_range = Self::calculate_value_range(expression, &value_ranges);
            let mut num_influenced_value =
                Self::calculate_num_influenced_values(expression, &num_influenced_values);
            let mut sensitivity = self.calculate_sensitivity(
                expression,
                &sensitivities,
                &value_range,
                num_influenced_value,
            );

            if self.output(sr).unwrap().eval()[0].condition.is_some()
                || self
                    .output(sr)
                    .unwrap()
                    .spawn()
                    .is_some_and(|s| s.condition.is_some())
            {
                // now the timing starts leaking information, so we have to disallow adding cutpoints
                sensitivity = SensitivityBound::Unbounded;
                num_influenced_value = NumInfluencedValues::Unbounded;
            }

            value_ranges.insert(sr, value_range);
            sensitivities.insert(sr, sensitivity);
            num_influenced_values.insert(sr, num_influenced_value);
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
            },
            |_, e| *e,
        );

        Ok(graph)
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
            ExpressionKind::StreamAccess(sr, StreamAccessKind::BoundedHold(n), _) => {
                *sensitivities
                    .get(sr)
                    .expect("dependencies should already be processed")
                    * (*n as u64)
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
                match w.aggr.op {
                    WindowOperation::Count => SensitivityBound::from(0.0),
                    WindowOperation::Sum
                    | WindowOperation::Conjunction
                    | WindowOperation::Disjunction => {
                        *sensitivities
                            .get(sr)
                            .expect("dependencies should already be processed")
                            * factor
                    }
                    _ => SensitivityBound::Unbounded,
                }
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
            ExpressionKind::Function(f) => match f.name.as_str() {
                "cast" => self.calculate_sensitivity(
                    &f.args[0],
                    sensitivities,
                    value_range,
                    num_influenced_values,
                ),
                _ => SensitivityBound::Unbounded,
            },
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
                StreamAccessKind::Sync
                | StreamAccessKind::Offset(_)
                | StreamAccessKind::Hold
                | StreamAccessKind::BoundedHold(_),
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
            ExpressionKind::Function(f) => match f.name.as_str() {
                "cast" => Self::calculate_value_range(&f.args[0], value_ranges),
                _ => ValueRange::Unbounded,
            },
            ExpressionKind::Ite {
                condition: _,
                consequence,
                alternative,
            } => {
                let cons_range = Self::calculate_value_range(consequence, value_ranges);
                let alt_range = Self::calculate_value_range(alternative, value_ranges);
                cons_range.union(alt_range)
            }
            _ => ValueRange::Unbounded,
        }
    }

    fn calculate_num_influenced_values(
        expression: &Expression,
        num_influenced_values: &HashMap<SRef, NumInfluencedValues>,
    ) -> NumInfluencedValues {
        match &expression.kind {
            // the parameter is not private because we don't have a spawn condition (otherwise num_influenced_values is set to Unbounded)
            ExpressionKind::ParameterAccess(_, _) => NumInfluencedValues::Bounded(0),
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
            ) => num_influenced_values[target],
            ExpressionKind::StreamAccess(target, StreamAccessKind::BoundedHold(n), _) => {
                match num_influenced_values[target] {
                    NumInfluencedValues::Bounded(b) => NumInfluencedValues::Bounded(b * *n),
                    NumInfluencedValues::Unbounded => NumInfluencedValues::Unbounded,
                }
            }
            // functions only ever operate on a single timestamps
            ExpressionKind::Function(f) => f
                .args
                .iter()
                .map(|e| Self::calculate_num_influenced_values(e, num_influenced_values))
                .reduce(|a, b| a + b)
                .unwrap(),
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => Self::calculate_num_influenced_values(consequence, num_influenced_values).max(
                Self::calculate_num_influenced_values(alternative, num_influenced_values)
                    + Self::calculate_num_influenced_values(condition, num_influenced_values),
            ),
            ExpressionKind::Default { expr, default } => {
                Self::calculate_num_influenced_values(expr, num_influenced_values)
                    + Self::calculate_num_influenced_values(default, num_influenced_values)
            }
            _ => NumInfluencedValues::Unbounded,
        }
    }

    fn add_noise_to_inputs(&mut self, inputs: &[(InputReference, f64)]) {
        let mut mapping = HashMap::new();
        let mut exclude = HashSet::new();
        let mut noise_additions = Vec::new();
        for (input, amount) in inputs {
            let input_id = self.next_expr_id();
            let input = &self.inputs[*input];
            let expr = Expression {
                kind: ExpressionKind::StreamAccess(input.sr, StreamAccessKind::Sync, vec![]),
                eid: input_id,
                span: Span::Unknown,
            };
            assert!(self
                .expr_maps
                .exprid_to_expr
                .insert(input_id, expr)
                .is_none());
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
            mapping.insert(input.sr, sr);
            exclude.insert(input_id);
            noise_additions.push((sr, amount));
        }
        self.replace_srs(mapping, exclude);
        for (sr, amount) in noise_additions {
            self.add_noise(sr, *amount);
        }
    }

    fn add_noise(&mut self, stream: SRef, amount: f64) {
        match stream {
            StreamReference::In(idx) => {
                self.add_noise_to_inputs(&[(idx, amount)]);
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
                assert!(self
                    .expr_maps
                    .exprid_to_expr
                    .insert(amount_id, amount_expr.clone())
                    .is_none(),);

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
                assert!(self
                    .expr_maps
                    .exprid_to_expr
                    .insert(noise_expr_id, noise_expr.clone())
                    .is_none());

                let new_expr_id = self.next_expr_id();
                let new_expr = Expression {
                    kind: ExpressionKind::ArithLog(
                        ArithLogOp::Add,
                        vec![self.expr_maps.exprid_to_expr[&old_expr].clone(), noise_expr],
                    ),
                    eid: new_expr_id,
                    span: Span::Unknown,
                };
                assert!(self
                    .expr_maps
                    .exprid_to_expr
                    .insert(new_expr_id, new_expr)
                    .is_none());
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

    fn replace_srs(
        &mut self,
        mapping: HashMap<StreamReference, StreamReference>,
        exclude: HashSet<ExprId>,
    ) {
        for (id, expr) in self.expr_maps.exprid_to_expr.iter_mut() {
            if exclude.contains(id) {
                continue;
            }
            Self::replace_sr_prime(expr, &mapping);
        }
    }

    fn replace_sr_prime(
        expr: &mut Expression,
        mapping: &HashMap<StreamReference, StreamReference>,
    ) {
        match &mut expr.kind {
            ExpressionKind::LoadConstant(_) => {}
            ExpressionKind::Tuple(expressions) | ExpressionKind::ArithLog(_, expressions) => {
                for expr in expressions {
                    Self::replace_sr_prime(expr, mapping);
                }
            }
            ExpressionKind::StreamAccess(stream_reference, _, expressions) => {
                if let Some(to) = mapping.get(stream_reference) {
                    *stream_reference = *to;
                }
                for expr in expressions {
                    Self::replace_sr_prime(expr, mapping);
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
                    Self::replace_sr_prime(expr, mapping);
                }
            }
            ExpressionKind::Function(fn_expr_kind) => {
                for expr in &mut fn_expr_kind.args {
                    Self::replace_sr_prime(expr, mapping);
                }
            }
            ExpressionKind::Widen(_) => todo!(),
            ExpressionKind::Default { expr, default } => {
                for expr in [expr, default] {
                    Self::replace_sr_prime(expr, mapping);
                }
            }
            ExpressionKind::TupleAccess(expression, _) => {
                Self::replace_sr_prime(expression, mapping);
            }
        }
    }

    fn get_input_annotations(
        &self,
    ) -> Result<(HashMap<SRef, SensitivityBound>, HashMap<SRef, ValueRange>), RtLolaError> {
        let mut sensitivities = HashMap::new();
        let mut value_ranges = HashMap::new();

        fn parse_bound(
            value: Option<&String>,
            span: Span,
            missing_msg: &str,
            parse_msg: &str,
        ) -> Result<f64, Diagnostic> {
            let v = match value {
                Some(v) => v,
                None => {
                    return Err(Diagnostic::error(missing_msg)
                        .add_span_with_label(span, Some("Missing value here".into()), true)
                        .into());
                }
            };
            match v.parse() {
                Ok(n) => Ok(n),
                Err(_) => Err(Diagnostic::error(parse_msg)
                    .add_span_with_label(span, Some("Faulty tag here".into()), true)
                    .into()),
            }
        }

        for input in &self.inputs {
            let value_range_from = input.tags.get("range_from");
            let value_range_to = input.tags.get("range_to");
            let value_range = match (value_range_from, value_range_to) {
                (Some(lower), Some(upper)) => {
                    let lower = parse_bound(
                        lower.value.as_ref(),
                        lower.span,
                        "Each range_from tag must have associated value",
                        "range_from tag must be a number",
                    )?;

                    let upper = parse_bound(
                        upper.value.as_ref(),
                        upper.span,
                        "Each range_to tag must have associated value",
                        "range_to tag must be a number",
                    )?;
                    ValueRange::Bounded {
                        lower: lower.try_into().unwrap(),
                        upper: upper.try_into().unwrap(),
                    }
                }
                (None, None) => ValueRange::Unbounded,
                _ => {
                    return Err(Diagnostic::error("Input stream must be either annotated with both `range_from` and `range_to` or neither.").add_span_with_label(input.span, Some("Found on this input stream"), true).into());
                }
            };

            value_ranges.insert(input.sr, value_range);

            if let ValueRange::Bounded { .. } = value_range {
                if input.tags.contains_key("sensitivity") {
                    return Err(Diagnostic::error("A stream was already tagged with a value range. Then a sensitivity should not be given.").add_span_with_label(input.tags.get("sensitivity").unwrap().span, Some("Found sensitivity tag here"), true).into());
                }
                let sensitivity = value_range.sensitivity();
                sensitivities.insert(input.sr, sensitivity);
            } else {
                let Some(sensitivity) = input.tags.get("sensitivity") else {
                    return Err(Diagnostic::error(
                        "Each input stream must be annotated with a sensitivity or value bound.",
                    )
                    .add_span_with_label(
                        input.span,
                        Some("Found input stream without annotations here"),
                        true,
                    )
                    .into());
                };
                let Some(sensitivity_value) = sensitivity.value.as_ref() else {
                    return Err(Diagnostic::error(
                        "Each sensitivity tag must have associated value.",
                    )
                    .add_span_with_label(
                        sensitivity.span,
                        Some("Found sensitivity tag without associated value here"),
                        true,
                    )
                    .into());
                };
                let sensitivity = if sensitivity_value == "unbounded" {
                    SensitivityBound::Unbounded
                } else {
                    let sensitivity: f64 = match sensitivity_value.parse() {
                        Ok(sens) => sens,
                        Err(_) => {
                            return Err(Diagnostic::error(
                                "A sensitivity tag must be either \"unbounded\" or a number.",
                            )
                            .add_span_with_label(
                                sensitivity.span,
                                Some("Found faulty sensitivity tag here."),
                                true,
                            )
                            .into());
                        }
                    };
                    sensitivity.into()
                };
                sensitivities.insert(input.sr, sensitivity);
            }
        }

        Ok((sensitivities, value_ranges))
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
    use std::{
        collections::{HashMap, HashSet},
        convert::TryInto,
    };

    use rtlola_parser::{parse, ParserConfig};

    use crate::{
        from_ast,
        hir::{DepAnaMode, DepAnaTrait, HirStage, StreamReference},
        modes::privacy::{SensitivityBound, ValueRange},
        RtLolaHir,
    };

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
        let graph = hir.analyze_dependency_graph(hir.graph().clone()).unwrap();
        let nodes: HashMap<_, _> = graph
            .node_indices()
            .map(|n| {
                let w = graph.node_weight(n).unwrap();
                (w.sref, w)
            })
            .collect();

        assert_eq!(
            nodes[&StreamReference::In(0)].sensitivity,
            SensitivityBound::from(5.0)
        );
        assert_eq!(
            nodes[&StreamReference::In(0)].value_range,
            ValueRange::Bounded {
                lower: 0.0f64.try_into().unwrap(),
                upper: 5.0f64.try_into().unwrap()
            }
        );

        assert_eq!(
            nodes[&StreamReference::Out(0)].sensitivity,
            SensitivityBound::from(5.0)
        );
        assert_eq!(
            nodes[&StreamReference::Out(0)].value_range,
            ValueRange::Bounded {
                lower: 1.0f64.try_into().unwrap(),
                upper: 6.0f64.try_into().unwrap()
            }
        );

        assert_eq!(
            nodes[&StreamReference::Out(1)].sensitivity,
            SensitivityBound::from(5.0)
        );
        assert_eq!(
            nodes[&StreamReference::Out(1)].value_range,
            ValueRange::Bounded {
                lower: 2.0f64.try_into().unwrap(),
                upper: 7.0f64.try_into().unwrap()
            }
        );

        assert_eq!(
            nodes[&StreamReference::Out(2)].sensitivity,
            SensitivityBound::from(10.0)
        );
        assert_eq!(
            nodes[&StreamReference::Out(2)].value_range,
            ValueRange::Bounded {
                lower: 3.0f64.try_into().unwrap(),
                upper: 13.0f64.try_into().unwrap()
            }
        );

        let cut_sets = RtLolaHir::<DepAnaMode>::enumerate_cuts(&graph, false);
        // assert_eq!(cut_sets.len(), 3);

        let cut_sets_as_srefs: Vec<HashSet<_>> = cut_sets
            .iter()
            .map(|cut_set| {
                cut_set
                    .iter()
                    .map(|n| graph.node_weight(*n).unwrap().sref)
                    .collect::<HashSet<_>>()
            })
            .collect();

        let expected: Vec<HashSet<_>> = vec![
            vec![StreamReference::In(0)].into_iter().collect(),
            vec![StreamReference::Out(0), StreamReference::Out(1)]
                .into_iter()
                .collect(),
            vec![StreamReference::Out(2)].into_iter().collect(),
        ];

        for cs in expected {
            assert!(
                cut_sets_as_srefs.iter().any(|cut_set| cut_set == &cs),
                "{:?} not in {:?}",
                cs,
                &cut_sets_as_srefs
            );
        }
    }
}
