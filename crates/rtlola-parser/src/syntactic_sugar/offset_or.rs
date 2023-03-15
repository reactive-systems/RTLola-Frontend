use super::{ChangeSet, SynSugar};
use crate::ast::{Expression, ExpressionKind, Offset, RtLolaAst};

/// Allows for using a offset(by: off, or: dft) function to assign an offset access a default value directly.
///
/// Transforms:
/// a.offset(by: off, or: dft) => a.offset(by: off).defaults(to: dft)
#[derive(Debug, Clone)]
pub(crate) struct OffsetOr {}

impl OffsetOr {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> ChangeSet {
        match &expr.kind {
            ExpressionKind::Method(base, name, _types, arguments) if "offset(by:or:)" == name.to_string() => {
                let target_stream = base.clone();
                assert_eq!(arguments.len(), 2);
                let offset = arguments[0].clone();
                let default = arguments[1].clone();
                let new_id = expr.id.primed();
                let new_access = Expression {
                    kind: ExpressionKind::Offset(
                        target_stream,
                        Offset::Discrete(offset.to_string().parse::<i16>().unwrap()),
                    ),
                    id: new_id,
                    span: expr.span,
                };
                let new_expr = Expression {
                    kind: ExpressionKind::Default(Box::new(new_access), Box::new(default)),
                    id: ast.next_id(),
                    span: expr.span,
                };
                ChangeSet::replace_current_expression(new_expr)
            },
            _ => ChangeSet::empty(),
        }
    }
}

impl SynSugar for OffsetOr {
    fn desugarize_expr<'a>(&self, exp: &'a Expression, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(exp, ast)
    }
}
