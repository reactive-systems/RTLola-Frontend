use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{EvalSpec, Mirror as AstMirror, Output, RtLolaAst};

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
        println!("Target: {}", target.is_some());
        let target = target.expect("mirror stream refers to a stream that does not exist");
        let target = (**target).clone();
        if target.eval.len() > 1 {
            // Do not apply mirror resolving until only a single eval statement remains
            return ChangeSet::empty();
        } else if target.eval.is_empty() {
            unimplemented!("Mirror stream defined on output without expression invalid")
        }
        let target_eval_spec = target.eval[0].to_owned();
        let filter_span = filter.span.clone();
        let new_eval_spec = EvalSpec {
            filter: Some(filter),
            id: ast.next_id(),
            span: Span::Indirect(Box::new(filter_span)),
            ..target_eval_spec
        };
        let output = Output {
            name,
            eval: vec![new_eval_spec],
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
