use super::{builder::Builder, ChangeSet, SynSugar};
use crate::ast::{EvalSpec, Mirror as AstMirror, Output, OutputKind, RtLolaAst};

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
            span: _,
            id: mirror_id,
        } = stream.clone();
        let target = ast
            .outputs
            .iter()
            .find(|o| o.name().is_some_and(|name| name.name == target.name))
            .expect("mirror stream refers to a stream that does not exist");
        let Output {
            kind: _,
            annotated_type,
            params,
            spawn,
            eval,
            close,
            tags,
            id,
            span,
        } = target.next_id(ast);

        let eval = eval
            .into_iter()
            .map(|e| {
                let EvalSpec {
                    annotated_pacing,
                    condition,
                    eval_expression,
                    id: _,
                    span,
                } = e;
                let builder = Builder::new(span, ast);

                let condition = match condition {
                    Some(condition) => builder.and(condition, filter.next_id(ast)),
                    None => filter.next_id(ast),
                };
                builder.eval_spec(Some(condition), annotated_pacing, eval_expression)
            })
            .collect();
        let output = Output {
            kind: OutputKind::NamedOutput(name),
            eval,
            id,
            span,
            annotated_type,
            params,
            spawn,
            close,
            tags,
        };
        ChangeSet::replace_stream(mirror_id, output)
    }
}

impl SynSugar for Mirror {
    fn desugarize_stream_mirror<'a>(&self, stream: &'a AstMirror, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(stream, ast)
    }
}
