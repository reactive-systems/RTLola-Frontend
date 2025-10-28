use rtlola_reporting::RtLolaError;

use super::{builder::Builder, ChangeSet, ExprOrigin, SynSugar};
use crate::ast::{Expression, ExpressionKind, RtLolaAst, WindowOperation};

/// Allows shorthand writing of aggregation windows as methods.
///
/// Transforms:
/// a.count(6s) => a.aggregate(using: count, over: 6s)
#[derive(Debug, Clone)]
pub(crate) struct AggrMethodToWindow {}

impl AggrMethodToWindow {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Method(base, name, _types, arguments)
                if name.arg_names.len() == 0
                    || (name.arg_names.len() == 1
                        && name.arg_names[0]
                            .as_ref()
                            .is_some_and(|ident| ident.name == "over")) =>
            {
                let aggregation = match name.name.name.as_ref() {
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
                    "ratio" => WindowOperation::TrueRatio,
                    "prob" | "probability" => WindowOperation::ConditionalProbability,
                    _ => return ChangeSet::empty(),
                };
                let builder = Builder::new(expr.span, ast);
                let target_stream = base.clone();
                let wait = false;
                let new_expr = if arguments.len() == 0 {
                    builder.all_aggregation(*target_stream, aggregation)
                } else {
                    let duration = arguments[0].clone();
                    builder.sliding_window(*target_stream, duration, wait, aggregation)
                };
                ChangeSet::replace_current_expression(new_expr)
            }
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for AggrMethodToWindow {
    fn desugarize_expr<'a>(
        &self,
        exp: &'a Expression,
        ast: &'a RtLolaAst,
        _stream: usize,
        _origin: ExprOrigin,
    ) -> Result<ChangeSet, RtLolaError> {
        Ok(self.apply(exp, ast))
    }
}
