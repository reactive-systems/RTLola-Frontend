use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{BinOp, Expression, ExpressionKind, Offset, RtLolaAst, StreamAccessKind};

/// Allows for using a delta(x,dft: 0)  function to compute the difference between the current and last value of x; defaults to 0.
///
/// Transforms:
/// delta(x,dft:0.0) → x - x.last(or: 0.0)
#[derive(Debug, Clone)]
pub(crate) struct Delta {}

impl Delta {
    fn apply<'a>(&self, expr: &Expression, ast: &'a RtLolaAst) -> ChangeSet {
        match &expr.kind {
            // Function(FunctionName, Vec<Type>, Vec<Expression>),
            ExpressionKind::Function(name, _types, args) => {
                let f_name = name.name.name.clone();
                /* currently not parsable but intended: , "δ", "Δ" */
                if !["delta"].contains(&f_name.as_str()) {
                    return ChangeSet::empty();
                }
                let arg_names = name.arg_names.clone();
                if arg_names.len() != 2 {
                    return ChangeSet::empty();
                }
                if arg_names[0].is_some() || arg_names[1].is_none() {
                    return ChangeSet::empty();
                }
                if !["dft", "default", "or"].contains(&arg_names[1].as_ref().unwrap().name.as_str()) {
                    return ChangeSet::empty();
                }
                let target_stream = args[0].clone();
                let new_id = expr.id.primed();

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
                let default_expr = args[1].clone();
                let default = Expression {
                    kind: ExpressionKind::Default(Box::new(offset), Box::new(default_expr)),
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
