use std::rc::Rc;

use rtlola_reporting::{Diagnostic, RtLolaError};

use super::{ChangeSet, ExprOrigin, SynSugar};
use crate::{
    ast::{
        AnnotatedPacingType, Expression, ExpressionKind, Ident, Literal, Output, OutputKind,
        RtLolaAst,
    },
    syntactic_sugar::builder::Builder,
};

/// Allows for using prob(of: x, given: y) for calculating (conditional) probabilities.
///
/// Transforms:
/// `prob(of: a, given: b, prior: p, confidence: c)` => `if (count_given' + 2.0) = 0.0 then 0.0 else (count_both' + 0.5 * 2.0) / (count_given' + 2.0)` and adds streams
/// ```lola
/// output count_both' eval with count_both'.offset(by: -1).defaults(to: 0.0) + if a ∧ b then 1.0 else 0.0
/// output count_given' eval with count_given'.offset(by: -1).defaults(to: 0.0) + if b ∧ a = a then 1.0 else 0.0
/// ```
#[derive(Debug, Clone)]
pub(crate) struct Probability {}

impl Probability {
    fn apply(
        &self,
        expr: &Expression,
        ast: &RtLolaAst,
        stream: usize,
        origin: ExprOrigin,
    ) -> Result<ChangeSet, RtLolaError> {
        let cs = match &expr.kind {
            ExpressionKind::Function(name, _types, arguments)
                if name.name.name.as_str() == "prob" =>
            {
                let builder = Builder::new(expr.span, ast);

                let (of_expr, given_expr, prior_expr, confidence_expr) =
                    match name.to_string().as_str() {
                        "prob(of:)" => (&arguments[0], None, None, None),
                        "prob(of:given:)" => (&arguments[0], Some(&arguments[1]), None, None),
                        "prob(of:given:prior:confidence:)" => (
                            &arguments[0],
                            Some(&arguments[1]),
                            Some(&arguments[2]),
                            Some(&arguments[3]),
                        ),
                        name => {
                            return Err(Diagnostic::error(&format!(
                                "Unsupported arguments to \"prob\" function: {name}",
                            ))
                            .add_span_with_label(expr.span, Some("Found prob function here"), true)
                            .into())
                        }
                    };

                let eval_clause = match origin {
                    ExprOrigin::EvalWith(c) => c,
                    ExprOrigin::SpawnWhen
                    | ExprOrigin::SpawnWith
                    | ExprOrigin::EvalWhen(_)
                    | ExprOrigin::CloseWhen => {
                        return Err(Diagnostic::error(
                            "Prob functions are only supported in eval with clauses",
                        )
                        .add_span_with_label(expr.span, Some("Found unsupported prob here."), true)
                        .into())
                    }
                };

                let const_1 = builder.literal(Literal::new_numeric(
                    ast.next_id(),
                    "1.0",
                    None,
                    builder.span.to_indirect(),
                ));
                let const_0 = builder.literal(Literal::new_numeric(
                    ast.next_id(),
                    "0.0",
                    None,
                    builder.span.to_indirect(),
                ));

                let stream = &ast.outputs[stream];
                let spawn = stream.spawn.as_ref();
                let filter = stream.eval[eval_clause].condition.as_ref();
                let close = stream.close.as_ref();
                let params = &stream.params;
                let param_exprs = params
                    .iter()
                    .map(|p| builder.ident(p.name.next_id(ast)))
                    .collect::<Vec<_>>();

                let count_both_ident =
                    Ident::new(ast.primed_name("count_both"), expr.span.to_indirect());
                let last_count_both = builder.last(
                    count_both_ident.next_id(ast),
                    param_exprs.iter().map(|e| e.next_id(ast)).collect(),
                    const_0.next_id(ast),
                );

                let count_both = if let Some(given_expr) = given_expr {
                    builder.if_then_else(
                        builder.and(of_expr.next_id(ast), given_expr.next_id(ast)),
                        const_1.next_id(ast),
                        const_0.next_id(ast),
                    )
                } else {
                    builder.if_then_else(
                        of_expr.next_id(ast),
                        const_1.next_id(ast),
                        const_0.next_id(ast),
                    )
                };
                let count_both_expr = builder.add(last_count_both, count_both);
                let count_both_stream = Output {
                    kind: OutputKind::NamedOutput(count_both_ident.next_id(ast)),
                    annotated_type: None,
                    params: params.iter().map(|p| Rc::new(p.next_id(ast))).collect(),
                    spawn: spawn.map(|s| s.next_id(ast)),
                    eval: vec![builder.eval_spec(
                        filter.map(|f| f.next_id(ast)),
                        AnnotatedPacingType::NotAnnotated(builder.span.to_indirect()),
                        Some(count_both_expr),
                    )],
                    close: close.map(|c| c.next_id(ast)),
                    tags: Vec::new(),
                    id: ast.next_id(),
                    span: expr.span.to_indirect(),
                };

                let count_given_ident =
                    Ident::new(ast.primed_name("count_given"), expr.span.to_indirect());
                let count_given_cond = if let Some(given_expr) = given_expr {
                    builder.and(
                        given_expr.next_id(ast),
                        builder.eq(of_expr.next_id(ast), of_expr.next_id(ast)),
                    )
                } else {
                    builder.eq(of_expr.next_id(ast), of_expr.next_id(ast))
                };
                let count_given = builder.if_then_else(
                    count_given_cond,
                    const_1.next_id(ast),
                    const_0.next_id(ast),
                );
                let last_count_given = builder.last(
                    count_given_ident.next_id(ast),
                    param_exprs
                        .iter()
                        .map(|p| p.next_id(ast))
                        .collect::<Vec<_>>(),
                    const_0.next_id(ast),
                );
                let count_given_expr = builder.add(last_count_given, count_given);
                let count_given_stream = Output {
                    kind: OutputKind::NamedOutput(count_given_ident.next_id(ast)),
                    annotated_type: None,
                    params: params.iter().map(|p| Rc::new(p.next_id(ast))).collect(),
                    spawn: spawn.map(|s| s.next_id(ast)),
                    eval: vec![builder.eval_spec(
                        filter.map(|f| f.next_id(ast)),
                        AnnotatedPacingType::NotAnnotated(builder.span.to_indirect()),
                        Some(count_given_expr),
                    )],
                    close: close.map(|c| c.next_id(ast)),
                    tags: Vec::new(),
                    id: ast.next_id(),
                    span: expr.span.to_indirect(),
                };

                let denom = if let Some(confidence) = confidence_expr {
                    builder.parentesized(builder.add(
                        builder.sync(
                            count_given_ident,
                            param_exprs.iter().map(|p| p.next_id(ast)).collect(),
                        ),
                        confidence.next_id(ast),
                    ))
                } else {
                    builder.sync(
                        count_given_ident,
                        param_exprs.iter().map(|p| p.next_id(ast)).collect(),
                    )
                };

                let numerator = if let Some(prior) = prior_expr {
                    builder.parentesized(builder.add(
                        builder.sync(
                            count_both_ident,
                            param_exprs.iter().map(|p| p.next_id(ast)).collect(),
                        ),
                        builder.mul(prior.next_id(ast), confidence_expr.unwrap().next_id(ast)),
                    ))
                } else {
                    builder.sync(
                        count_both_ident,
                        param_exprs.iter().map(|p| p.next_id(ast)).collect(),
                    )
                };

                let prob_stream_access = builder.div(numerator, denom.next_id(ast));

                let prob_stream_access = builder.if_then_else(
                    builder.eq(denom.next_id(ast), const_0.next_id(ast)),
                    const_0.next_id(ast),
                    prob_stream_access,
                );
                ChangeSet::add_output(count_given_stream)
                    + ChangeSet::add_output(count_both_stream)
                    + ChangeSet::replace_current_expression(prob_stream_access)
            }
            _ => ChangeSet::empty(),
        };
        Ok(cs)
    }
}

impl SynSugar for Probability {
    fn desugarize_expr<'a>(
        &self,
        exp: &'a Expression,
        ast: &'a RtLolaAst,
        stream: usize,
        origin: ExprOrigin,
    ) -> Result<ChangeSet, RtLolaError> {
        self.apply(exp, ast, stream, origin)
    }
}
