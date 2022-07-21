use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{BinOp, EvalSpec, Expression, Mirror as AstMirror, Output, RtLolaAst};

/// Enables usage of mirror streams
///
/// Transforms:
/// output a mirrors b when φ → output a filter when φ := b
#[derive(Debug, Clone)]
pub(crate) struct Mirror {}

impl Mirror {
    fn apply<'a>(&self, stream: &'a AstMirror, ast: &'a RtLolaAst) -> ChangeSet {
        let AstMirror {
            name,
            target,
            filter,
            span,
            id: mirror_id,
        } = stream.clone();
        let target = ast.outputs.iter().find(|o| o.name.name == target.name);
        let target = target.expect("mirror stream refers to a stream that does not exist");
        let target = (**target).clone();

        let target_eval_specs = target.eval.clone();
        let filter_span = &filter.span;
        let new_eval_specs = target_eval_specs
            .into_iter()
            .map(|e| {
                let EvalSpec {
                    annotated_pacing: t_annotated_pacing,
                    condition: t_filter,
                    eval_expression: t_eval,
                    id: t_id,
                    span: t_span,
                } = e;

                let new_filter = match t_filter {
                    Some(old_f) => {
                        Expression {
                            id: ast.next_id(),
                            span: Span::Indirect(Box::new(filter_span.clone())),
                            kind: crate::ast::ExpressionKind::Binary(
                                BinOp::And,
                                Box::new(old_f),
                                Box::new(filter.clone()),
                            ),
                        }
                    },
                    None => filter.clone(),
                };
                EvalSpec {
                    condition: Some(new_filter),
                    id: t_id.primed(),
                    span: Span::Indirect(Box::new(t_span)),
                    annotated_pacing: t_annotated_pacing,
                    eval_expression: t_eval,
                }
            })
            .collect();
        let output = Output {
            name,
            eval: new_eval_specs,
            id: ast.next_id(),
            span: Span::Indirect(Box::new(span)),
            ..target
        };
        ChangeSet::replace_stream(mirror_id, output)
    }
}

impl SynSugar for Mirror {
    fn desugarize_stream_mirror<'a>(&self, stream: &'a AstMirror, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(stream, ast)
    }
}
