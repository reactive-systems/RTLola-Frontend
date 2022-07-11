use rtlola_reporting::Span;

use super::{ChangeSet, SynSugar};
use crate::ast::{BinOp, EvalSpec, Expression, Output, RtLolaAst};

/// Enables usage of multiple eval conditions
///
/// Transforms:
/// output a eval when c1 with o1 eval when c2 with o2 eval when c3 with o3
/// to:
/// output a eval when c1 || c2 || c3 eval with if c1 then o1 else if c2 then o2 else o3
///
/// If is ordered according to the eval cases, i.e., when multiple conditions are true
/// the one given first in the input string is used.
///
/// The condition of the last case in the evaluation is not tested and assumed to be active
/// through to the filter condition
#[derive(Debug, Clone)]
pub(crate) struct MultipleEval {}

impl MultipleEval {
    fn apply<'a>(&self, stream: &'a Output, ast: &'a RtLolaAst) -> ChangeSet {
        let Output {
            name, eval, id, span, ..
        } = stream.clone();
        if eval.len() <= 1 {
            // Nothing to do when only a single eval clause is given
            return ChangeSet::empty();
        } else if eval.is_empty() {
            unimplemented!("Desugarization for streams without eval clause incomplete")
        }

        let mut eval = eval;
        let last_eval_expr = eval.remove(eval.len() - 1);

        let new_eval_spec = eval.into_iter().rev().fold(last_eval_expr, |old, new| {
            let EvalSpec {
                annotated_pacing,
                filter,
                eval_expression,
                id,
                span,
            } = new;
            let EvalSpec {
                annotated_pacing: annotated_pacing_o,
                filter: filter_o,
                eval_expression: eval_expression_o,
                ..
            } = old;

            let next_pt = match (annotated_pacing_o, annotated_pacing) {
                (Some(_pto), Some(_ptn)) => {
                    // This case handles ors all pacing types but is not wanted or may result in annotated disjunctions and inferred conjunctions.
                    //let span = ptn.span.clone();
                    //Some(Expression{kind: crate::ast::ExpressionKind::Binary(BinOp::Or ,Box::new(ptn), Box::new(pto)), id:ast.next_id(), span})
                    unimplemented!("Multiple eval clauses cannot be annotated with multiple pacing types");
                },
                (Some(x), None) => Some(x),
                (None, Some(y)) => Some(y),
                (None, None) => None,
            };

            let (current_filter, next_filter) = match (filter_o, filter) {
                (Some(f), Some(f_new)) => {
                    let span = f_new.span.clone();
                    let next_filter = Some(Expression {
                        kind: crate::ast::ExpressionKind::Binary(BinOp::Or, Box::new(f_new.clone()), Box::new(f)),
                        id: ast.next_id(),
                        span,
                    });
                    (Some(f_new), next_filter)
                },
                (None, Some(x)) => (Some(x.clone()), Some(x)),
                (Some(y), None) => (None, Some(y)), //this case should not be reached
                (None, None) => (None, None),       // this case should not be reached
            };

            let current_filter = current_filter.unwrap();
            let next_eval = match (eval_expression_o, eval_expression) {
                (Some(eval_old), Some(eval_new)) => {
                    let span = eval_new.span.clone();
                    let expr = Expression {
                        kind: crate::ast::ExpressionKind::Ite(
                            Box::new(current_filter),
                            Box::new(eval_new),
                            Box::new(eval_old),
                        ),
                        id: ast.next_id(),
                        span,
                    };

                    Some(expr)
                },
                (_, _) => unimplemented!("Desugarize multiple eval clauses without eval expression"),
            };

            EvalSpec {
                annotated_pacing: next_pt,
                filter: next_filter,
                eval_expression: next_eval,
                id,
                span,
            }
        });

        let output = Output {
            name,
            eval: vec![new_eval_spec],
            id: ast.next_id(),
            span: Span::Indirect(Box::new(span)),
            ..stream.clone()
        };
        ChangeSet::replace_stream(id, output)
    }
}

impl SynSugar for MultipleEval {
    fn desugarize_stream_out<'a>(&self, stream: &'a Output, ast: &'a RtLolaAst) -> ChangeSet {
        self.apply(stream, ast)
    }
}
