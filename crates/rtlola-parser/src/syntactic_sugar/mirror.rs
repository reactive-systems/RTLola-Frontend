use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{FilterSpec, Mirror as AstMirror, Output, RtLolaAst};

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
        let filter_span = filter.span.clone();
        let output = Output {
            name,
            filter: Some(FilterSpec {
                target: filter,
                id: ast.next_id(),
                span: Span::Indirect(Box::new(filter_span)),
            }),
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
