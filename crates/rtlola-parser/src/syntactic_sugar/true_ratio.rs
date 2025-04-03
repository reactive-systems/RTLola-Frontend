use std::rc::Rc;

use rtlola_reporting::{Diagnostic, RtLolaError};

use crate::{
    ast::{
        AnnotatedPacingType, Expression, ExpressionKind, FunctionName, Ident, InstanceOperation,
        Literal, Output, WindowOperation,
    },
    syntactic_sugar::builder::Builder,
    RtLolaAst,
};

use super::{ChangeSet, ExprOrigin, SynSugar};

/// Allows for using the method 'true_ratio' in an aggregation method.
///
/// Transforms:
/// a.aggregate(over:x, using: true_ratio) => output a' := if a then 1.0 else 0.0
///                                    a'.aggregate(over:x, using: avg)
#[derive(Debug, Clone)]
pub(crate) struct TrueRatio {}

impl TrueRatio {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> Result<ChangeSet, RtLolaError> {
        let target_stream = match &expr.kind {
            ExpressionKind::SlidingWindowAggregation {
                expr: target_stream,
                aggregation: WindowOperation::TrueRatio,
                ..
            }
            | ExpressionKind::InstanceAggregation {
                expr: target_stream,
                aggregation: InstanceOperation::TrueRatio,
                ..
            } => target_stream,
            _ => return Ok(ChangeSet::empty()),
        };
        let builder = Builder::new(expr.span, ast);
        let ident_of_target_stream = match &target_stream.kind {
            ExpressionKind::Ident(ident) => ident,
            ExpressionKind::Function(name, ty, _para) if ty.is_empty() => &name.name,
            k => {
                return Err(Diagnostic::error(&format!(
                    "Found aggregation over unsupported expression kind: {k:?}"
                ))
                .add_span_with_label(
                    target_stream.span,
                    Some("found unsupported expression here"),
                    true,
                )
                .into())
            }
        };
        let primed_ident = Ident::new(
            ast.primed_name(&ident_of_target_stream.name),
            ident_of_target_stream.span.to_indirect(),
        );
        let primed_access = match &target_stream.kind {
            ExpressionKind::Ident(_ident) => builder.ident(primed_ident.next_id(ast)),
            ExpressionKind::Function(f_name, ty, para) if ty.is_empty() => builder.function(
                FunctionName {
                    name: primed_ident.next_id(ast),
                    arg_names: f_name
                        .arg_names
                        .iter()
                        .map(|arg| arg.as_ref().map(|a| a.next_id(ast)))
                        .collect(),
                },
                ty.iter().map(|ty| ty.next_id(ast)).collect(),
                para.iter().map(|p| p.next_id(ast)).collect(),
            ),
            _ => unreachable!("would have aborted above"),
        };
        let new_expr = match &expr.kind {
            ExpressionKind::SlidingWindowAggregation {
                expr: _,
                duration,
                wait,
                aggregation: _,
            } => builder.sliding_window(
                primed_access,
                duration.next_id(ast),
                *wait,
                WindowOperation::Average,
            ),
            ExpressionKind::InstanceAggregation {
                expr: _,
                selection,
                aggregation: _,
            } => builder.instance_aggregation(
                primed_access,
                selection.next_id(ast),
                InstanceOperation::Average,
            ),
            _ => unreachable!(),
        };
        let const_1 = Literal::new_numeric(ast.next_id(), "1.0", None, builder.span.to_indirect());
        let const_0 = Literal::new_numeric(ast.next_id(), "0.0", None, builder.span.to_indirect());
        let output_kind = crate::ast::OutputKind::NamedOutput(primed_ident);

        let new_output = if let Some(input) = ast
            .inputs
            .iter()
            .find(|i| i.name.name == ident_of_target_stream.name)
        {
            let cond = target_stream.next_id(ast);
            let eval_expr =
                builder.if_then_else(cond, builder.literal(const_1), builder.literal(const_0));
            Output {
                kind: output_kind,
                annotated_type: None,
                params: Vec::new(),
                spawn: None,
                eval: vec![builder.eval_spec(
                    None,
                    AnnotatedPacingType::NotAnnotated(builder.span.to_indirect()),
                    Some(eval_expr),
                )],
                close: None,
                tags: Vec::new(),
                id: ast.next_id(),
                span: input.span.to_indirect(),
            }
        } else if let Some(output) = ast.outputs.iter().find(|o| {
            o.name()
                .is_some_and(|name| name.name == ident_of_target_stream.name)
        }) {
            let cond = match expr.kind {
                ExpressionKind::SlidingWindowAggregation { .. } => target_stream.next_id(ast),
                ExpressionKind::InstanceAggregation { .. } => builder.function(
                    FunctionName {
                        name: ident_of_target_stream.next_id(ast),
                        arg_names: vec![None; output.params.len()],
                    },
                    Vec::new(),
                    output
                        .params
                        .iter()
                        .map(|p| builder.ident(p.name.next_id(ast)))
                        .collect(),
                ),
                _ => unreachable!(),
            };
            let eval_expr =
                builder.if_then_else(cond, builder.literal(const_1), builder.literal(const_0));
            Output {
                kind: output_kind,
                annotated_type: None,
                params: output
                    .params
                    .iter()
                    .map(|p| Rc::new(p.next_id(ast)))
                    .collect(),
                spawn: output.spawn.as_ref().map(|s| s.next_id(ast)),
                eval: output
                    .eval
                    .iter()
                    .map(|spec| {
                        builder.eval_spec(
                            spec.condition.as_ref().map(|c| c.next_id(ast)),
                            spec.annotated_pacing.next_id(ast),
                            Some(eval_expr.next_id(ast)),
                        )
                    })
                    .collect(),
                close: output.close.as_ref().map(|c| c.next_id(ast)),
                tags: Vec::new(),
                id: ast.next_id(),
                span: output.span.to_indirect(),
            }
        } else {
            unimplemented!("True ratio over an input or output stream")
        };
        Ok(ChangeSet::add_output(new_output) + ChangeSet::replace_current_expression(new_expr))
    }
}

impl SynSugar for TrueRatio {
    fn desugarize_expr<'a>(
        &self,
        exp: &'a Expression,
        ast: &'a RtLolaAst,
        _stream: usize,
        _origin: ExprOrigin,
    ) -> Result<ChangeSet, RtLolaError> {
        self.apply(exp, ast)
    }
}
