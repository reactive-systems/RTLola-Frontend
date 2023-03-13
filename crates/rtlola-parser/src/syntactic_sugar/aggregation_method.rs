use super::{ChangeSet, SynSugar};
use crate::ast::{Expression, ExpressionKind, RtLolaAst, WindowOperation};

/// Allows shorthand writing of aggregation windows as methods.
///
/// Transforms:
/// a.count(6s) => a.aggregate(using: count, over: 6s)
#[derive(Debug, Clone)]
pub(crate) struct AggrMethodToWindow {}

impl AggrMethodToWindow {
    fn apply(&self, expr: &Expression) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Method(base, name, _types, arguments) => {
                let op = match name.name.name.as_ref() {
                    "count" => WindowOperation::Count,
                    "min" => WindowOperation::Min,
                    "max" => WindowOperation::Max,
                    "sum" => WindowOperation::Sum,
                    "avg" => WindowOperation::Average,
                    "integral" => WindowOperation::Integral,
                    "var" => WindowOperation::Variance,
                    "cov" => WindowOperation::Covariance,
                    "sd" => WindowOperation::StandardDeviation,
                    "med" => WindowOperation::NthPercentile(50),
                    _ => return ChangeSet::empty(),
                };
                let target_stream = base.clone();
                let wait = false;
                let duration = Box::new(arguments[0].clone());
                let new_id = expr.id.primed();
                let new_expr = Expression {
                    kind: ExpressionKind::SlidingWindowAggregation {
                        expr: target_stream,
                        duration,
                        wait,
                        aggregation: op,
                    },
                    id: new_id,
                    span: expr.span,
                };
                ChangeSet::replace_current_expression(new_expr)
            },
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for AggrMethodToWindow {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, _ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp)
    }
}
