use rtlola_reporting::RtLolaError;

use super::{builder::Builder, ChangeSet, ExprOrigin, SynSugar};
use crate::ast::{BinOp, Expression, ExpressionKind, RtLolaAst};

/// Allows for using a implies b.
///
/// Transforms:
/// a -> b => ¬a ∨ b
#[derive(Debug, Clone)]
pub(crate) struct Implication {}

impl Implication {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Binary(BinOp::Implies, lhs, rhs) => {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let builder = Builder::new(expr.span, ast);
                let lhs = match lhs.kind {
                    ExpressionKind::Lit(_)
                    | ExpressionKind::StreamAccess(_, _)
                    | ExpressionKind::Unary(_, _)
                    | ExpressionKind::DiscreteWindowAggregation { .. }
                    | ExpressionKind::SlidingWindowAggregation { .. }
                    | ExpressionKind::InstanceAggregation { .. }
                    | ExpressionKind::Function(_, _, _)
                    | ExpressionKind::Method(_, _, _, _)
                    | ExpressionKind::Ident(_)
                    | ExpressionKind::Default(_, _)
                    | ExpressionKind::Offset(_, _)
                    | ExpressionKind::Tuple(_)
                    | ExpressionKind::Field(_, _)
                    | ExpressionKind::ParenthesizedExpression(_)
                    | ExpressionKind::MissingExpression => lhs,
                    ExpressionKind::Binary(_, _, _)
                    | ExpressionKind::Ite(_, _, _)
                    | ExpressionKind::Lambda { .. } => Box::new(builder.parentesized(*lhs)),
                };
                let new_expr = builder.or(builder.not(*lhs), *rhs);
                ChangeSet::replace_current_expression(new_expr)
            }
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for Implication {
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
