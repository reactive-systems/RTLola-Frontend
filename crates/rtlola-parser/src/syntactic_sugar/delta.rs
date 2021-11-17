use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{BinOp, Expression, ExpressionKind, Literal, Offset, RtLolaAst, StreamAccessKind};

/// Allows for using a δ(x) function to compute the difference between the current and last value of x; defaults to 0.
///
/// Transforms:
/// δ(x) → x - x.last(or: x)
#[derive(Debug, Clone)]
pub(crate) struct Delta {}

impl Delta {
    fn apply<'a>(&self, expr: &Expression, ast: &'a RtLolaAst) -> ChangeSet {
        match &expr.kind {
            // Function(FunctionName, Vec<Type>, Vec<Expression>),
            ExpressionKind::Function(name, _types, args) => {
                if !["delta", "δ", "Δ"].contains(&name.name.name.as_str()) {
                    return ChangeSet::empty();
                }
                let target_stream = args[0].clone();
                let mut new_id = expr.id;
                new_id.prime_counter += 1;

                let indirect_span = Span::Indirect(Box::new(expr.span.clone()));

                let sync = Expression {
                    kind: ExpressionKind::StreamAccess(Box::new(target_stream.clone()), StreamAccessKind::Sync),
                    id: ast.next_id(),
                    span: expr.span.clone(),
                };
                let offset = Expression {
                    kind: ExpressionKind::Offset(Box::new(target_stream), Offset::Discrete(-1)),
                    id: new_id,
                    span: expr.span.clone(),
                };
                let zero = Expression {
                    kind: ExpressionKind::Lit(Literal::new_numeric(ast.next_id(), "0", None, indirect_span.clone())),
                    id: ast.next_id(),
                    span: indirect_span.clone(),
                };
                let default = Expression {
                    kind: ExpressionKind::Default(Box::new(offset), Box::new(zero)),
                    id: ast.next_id(),
                    span: indirect_span.clone(),
                };
                let res = Expression {
                    kind: ExpressionKind::Binary(BinOp::Sub, Box::new(sync), Box::new(default)),
                    id: new_id,
                    span: indirect_span,
                };
                ChangeSet::replace_current_expression(res)
            },
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for Delta {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp, ast)
    }
}
