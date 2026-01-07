use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    ops::{Add, Mul, Sub},
};

use num::ToPrimitive;
use ordered_float::NotNan;
use petgraph::{algo::toposort, prelude::StableGraph};
use rtlola_parser::ast::Tag;
use rtlola_reporting::{RtLolaError, Span};

use crate::hir::{
    ArithLogOp, Constant, DepAnaMode, DepAnaTrait, EdgeWeight, Expression, ExpressionKind, Hir,
    Inlined, Literal, SRef, StreamAccessKind, StreamReference,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum PrivacyHeuristic {
    Inputs,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
enum SensitivityBound {
    Bounded(NotNan<f64>),
    Unbounded,
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

#[derive(Clone, Copy, Debug)]
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
                lower: l1 - l2,
                upper: u1 - u2,
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

struct AugmentedNode {
    sref: SRef,
    sensitivity: SensitivityBound,
    value_range: ValueRange,
}

impl AugmentedNode {
    fn as_tags(&self) -> HashMap<String, Tag> {
        vec![
            (
                "derived_sensitivity".to_string(),
                Tag {
                    key: "derived_sensitivity".into(),
                    value: Some(self.sensitivity.to_string()),
                    span: Span::Unknown,
                },
            ),
            (
                "derived_value_range".to_string(),
                Tag {
                    key: "derived_value_range".into(),
                    value: Some(self.value_range.to_string()),
                    span: Span::Unknown,
                },
            ),
        ]
        .into_iter()
        .collect()
    }
}

type AugmentedDependencyGraph = StableGraph<AugmentedNode, EdgeWeight>;

impl Hir<DepAnaMode> {
    pub(crate) fn add_privacy_barriers(
        mut self,
        parameters: f64,
        heuristic: PrivacyHeuristic,
    ) -> Result<Self, RtLolaError> {
        let graph = self.analyze_dependency_graph();

        // debug annotations
        for node in graph.node_indices() {
            let node = graph.node_weight(node).unwrap();
            match node.sref {
                StreamReference::In(i) => self.inputs[i].tags.extend(node.as_tags()),
                StreamReference::Out(o) => self.outputs[o].tags.extend(node.as_tags()),
            }
        }

        Ok(self)
    }

    fn analyze_dependency_graph(&self) -> AugmentedDependencyGraph {
        let graph = self.graph();

        let (mut sensitivities, mut value_ranges) = self.get_input_annotations();
        let mut num_influenced_values: HashMap<_, _> = self
            .inputs
            .iter()
            .map(|i| (i.sr, NumInfluencedValues::Bounded(1)))
            .collect();

        for node in toposort(graph, None).expect("no cycles").into_iter().rev() {
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
            let sensitivity = Self::calculate_sensitivity(
                expression,
                &sensitivities,
                &value_range,
                num_influenced_value,
            );
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

        graph
    }

    fn calculate_sensitivity(
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
                let lhs_sens = Self::calculate_sensitivity(
                    lhs,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                let rhs_sens = Self::calculate_sensitivity(
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
            ExpressionKind::Default { expr, default } => {
                let expr_bound = Self::calculate_sensitivity(
                    expr,
                    sensitivities,
                    value_range,
                    num_influenced_values,
                );
                let default_bound = Self::calculate_sensitivity(
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
                Literal::Str(_) => unimplemented!(),
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
            ExpressionKind::StreamAccess(target, _, _) => value_ranges[target],
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
}
