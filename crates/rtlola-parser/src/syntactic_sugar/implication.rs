use super::{ChangeSet, SynSugar};
use crate::ast::{BinOp, Expression, ExpressionKind, Parenthesis, RtLolaAst, UnOp};

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
                let new_id = expr.id.primed();
                let lhs = match lhs.kind {
                    ExpressionKind::Lit(_)
                    | ExpressionKind::StreamAccess(_, _)
                    | ExpressionKind::Unary(_, _)
                    | ExpressionKind::DiscreteWindowAggregation {
                        expr: _,
                        duration: _,
                        wait: _,
                        aggregation: _,
                    }
                    | ExpressionKind::SlidingWindowAggregation {
                        expr: _,
                        duration: _,
                        wait: _,
                        aggregation: _,
                    }
                    | ExpressionKind::InstanceAggregation { .. }
                    | ExpressionKind::Function(_, _, _)
                    | ExpressionKind::Method(_, _, _, _)
                    | ExpressionKind::Ident(_)
                    | ExpressionKind::Default(_, _)
                    | ExpressionKind::Offset(_, _)
                    | ExpressionKind::Tuple(_)
                    | ExpressionKind::Field(_, _)
                    | ExpressionKind::ParenthesizedExpression(_, _, _)
                    | ExpressionKind::MissingExpression => lhs,
                    ExpressionKind::Binary(_, _, _) | ExpressionKind::Ite(_, _, _) => {
                        let lhs = Expression {
                            kind: ExpressionKind::ParenthesizedExpression(
                                Some(Box::new(Parenthesis {
                                    id: ast.next_id(),
                                    span: expr.span.to_indirect(),
                                })),
                                lhs,
                                Some(Box::new(Parenthesis {
                                    id: ast.next_id(),
                                    span: expr.span.to_indirect(),
                                })),
                            ),
                            id: new_id,
                            span: expr.span.to_indirect(),
                        };
                        Box::new(lhs)
                    },
                };
                let lhs = Expression {
                    kind: ExpressionKind::Unary(UnOp::Not, lhs),
                    id: ast.next_id(),
                    span: expr.span.to_indirect(),
                };
                let new_expr = Expression {
                    kind: ExpressionKind::Binary(BinOp::Or, Box::new(lhs), rhs),
                    id: ast.next_id(),
                    span: expr.span.to_indirect(),
                };
                ChangeSet::replace_current_expression(new_expr)
            },
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for Implication {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp, ast)
    }
}
