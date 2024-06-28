use super::{ChangeSet, SynSugar};
use crate::ast::{Expression, ExpressionKind, Offset, RtLolaAst};

/// Allows for using a last(or:) function to access an element with offset -1.
///
/// Transforms:
/// a.last(or: x) => a.offset(by: -1).defaults(to: x)
#[derive(Debug, Clone)]
pub(crate) struct Last {}

impl Last {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Method(base, name, _types, arguments) => {
                if "last(or:)" != name.to_string() {
                    return ChangeSet::empty();
                };
                let target_stream = base.clone();
                assert_eq!(arguments.len(), 1);
                let default = arguments[0].clone();
                let new_id = expr.id.primed();
                let new_access = Expression {
                    kind: ExpressionKind::Offset(target_stream, Offset::Discrete(-1)),
                    id: new_id,
                    span: expr.span.to_indirect(),
                };
                let new_expr = Expression {
                    kind: ExpressionKind::Default(Box::new(new_access), Box::new(default)),
                    id: ast.next_id(),
                    span: expr.span.to_indirect(),
                };
                ChangeSet::replace_current_expression(new_expr)
            },
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for Last {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp, ast)
    }
}