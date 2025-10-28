use rtlola_reporting::{Diagnostic, RtLolaError};

use super::{ChangeSet, ExprOrigin, SynSugar};
use crate::{
    ast::{AnnotatedPacingType, Expression, ExpressionKind, Ident, Output, OutputKind, RtLolaAst},
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
    #[cfg(not(feature = "probability"))]
    fn desugar_prob(
        expr: &Expression,
        ast: &RtLolaAst,
        origin: ExprOrigin,
        stream: usize,
        of_expr: &Expression,
        given_expr: Option<&Expression>,
        prior_expr: Option<&Expression>,
        confidence_expr: Option<&Expression>,
        duration_expr: Option<&Expression>,
    ) -> Result<ChangeSet, RtLolaError> {
        use crate::ast::Literal;
        use std::rc::Rc;
        assert!(duration_expr.is_none());
        let builder = Builder::new(expr.span, ast);

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
        let (spawn, filter, close) = match origin {
            ExprOrigin::EvalWith(c) => (
                stream.spawn.as_ref(),
                stream.eval[c].condition.as_ref(),
                stream.close.as_ref(),
            ),
            ExprOrigin::EvalWhen(_) => (stream.spawn.as_ref(), None, stream.close.as_ref()),
            ExprOrigin::SpawnWhen | ExprOrigin::SpawnWith | ExprOrigin::CloseWhen => {
                return Err(Diagnostic::error(
                    "Prob functions are only supported in eval with clauses",
                )
                .add_span_with_label(expr.span, Some("Found unsupported prob here."), true)
                .into())
            }
        };

        let params = &stream.params;
        let param_exprs = params
            .iter()
            .map(|p| builder.ident(p.name.next_id(ast)))
            .collect::<Vec<_>>();

        let count_both_ident = Ident::new(ast.primed_name("count_both"), expr.span.to_indirect());
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

        let count_given_ident = Ident::new(ast.primed_name("count_given"), expr.span.to_indirect());
        let count_given_cond = if let Some(given_expr) = given_expr {
            builder.and(
                given_expr.next_id(ast),
                builder.eq(of_expr.next_id(ast), of_expr.next_id(ast)),
            )
        } else {
            builder.eq(of_expr.next_id(ast), of_expr.next_id(ast))
        };
        let count_given =
            builder.if_then_else(count_given_cond, const_1.next_id(ast), const_0.next_id(ast));
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

        Ok(ChangeSet::add_output(count_given_stream)
            + ChangeSet::add_output(count_both_stream)
            + ChangeSet::replace_current_expression(prob_stream_access))
    }

    #[cfg(feature = "probability")]
    fn desugar_prob(
        expr: &Expression,
        ast: &RtLolaAst,
        _origin: ExprOrigin,
        _stream: usize,
        of_expr: &Expression,
        given_expr: Option<&Expression>,
        prior_expr: Option<&Expression>,
        confidence_expr: Option<&Expression>,
        duration_expr: Option<&Expression>,
    ) -> Result<ChangeSet, RtLolaError> {
        use crate::ast::WindowOperation;

        let builder = Builder::new(expr.span, ast);

        let target_stream_name = ast.primed_name("target");
        let target_stream_ident = Ident {
            name: target_stream_name,
            span: expr.span.to_indirect(),
        };
        let target_stream_expr = match (given_expr, prior_expr, confidence_expr) {
            (None, None, None) => of_expr.clone(),
            (Some(given_expr), None, None) => {
                builder.tuple(vec![of_expr.clone(), given_expr.clone()])
            }
            (Some(given_expr), Some(prior_expr), Some(confidence_expr)) => builder.tuple(vec![
                of_expr.clone(),
                given_expr.clone(),
                prior_expr.clone(),
                confidence_expr.clone(),
            ]),
            _ => unreachable!(),
        };

        let target_stream = Output {
            kind: OutputKind::NamedOutput(target_stream_ident.next_id(ast)),
            annotated_type: None,
            params: Vec::new(),
            spawn: None,
            eval: vec![builder.eval_spec(
                None,
                AnnotatedPacingType::NotAnnotated(builder.span.to_indirect()),
                Some(target_stream_expr),
            )],
            close: None,
            tags: Vec::new(),
            id: ast.next_id(),
            span: expr.span.to_indirect(),
        };

        let op = match (given_expr, prior_expr, confidence_expr) {
            (None, None, None) => WindowOperation::TrueRatio,
            (Some(_), None, None) => WindowOperation::ConditionalProbability,
            (Some(_), Some(_), Some(_)) => WindowOperation::ConditionalProbabilityWithPrior,
            _ => unreachable!(),
        };

        let expr = if let Some(duration_expr) = duration_expr {
            builder.sliding_window(
                builder.ident(target_stream_ident),
                duration_expr.next_id(ast),
                false,
                op,
            )
        } else {
            builder.all_aggregation(builder.ident(target_stream_ident), op)
        };

        Ok(ChangeSet::replace_current_expression(expr) + ChangeSet::add_output(target_stream))
    }

    fn apply(
        &self,
        expr: &Expression,
        ast: &RtLolaAst,
        stream: usize,
        origin: ExprOrigin,
    ) -> Result<ChangeSet, RtLolaError> {
        match &expr.kind {
            ExpressionKind::Method(of, name, _types, arguments)
                if name.name.name.as_str() == "prob" =>
            {
                let (of_expr, given_expr, prior_expr, confidence_expr, over_expr) =
                    match name.to_string().as_str() {
                        "prob(given:)" => (of, Some(&arguments[0]), None, None, None),
                        "prob(given:over:)" => {
                            (of, Some(&arguments[0]), None, None, Some(&arguments[0]))
                        }
                        "prob(given:prior:confidence:)" => (
                            of,
                            Some(&arguments[0]),
                            Some(&arguments[1]),
                            Some(&arguments[2]),
                            None,
                        ),
                        "prob(given:prior:confidence:over:)" => (
                            of,
                            Some(&arguments[0]),
                            Some(&arguments[1]),
                            Some(&arguments[2]),
                            Some(&arguments[3]),
                        ),
                        _ => return Ok(ChangeSet::empty()),
                    };
                Self::desugar_prob(
                    expr,
                    ast,
                    origin,
                    stream,
                    of_expr,
                    given_expr,
                    prior_expr,
                    confidence_expr,
                    over_expr,
                )
            }
            ExpressionKind::Function(name, _types, arguments)
                if name.name.name.as_str() == "prob" =>
            {
                let (of_expr, given_expr, prior_expr, confidence_expr, over_expr) =
                    match name.to_string().as_str() {
                        "prob(of:given:)" => (&arguments[0], Some(&arguments[1]), None, None, None),
                        "prob(of:given:over:)" => (
                            &arguments[0],
                            Some(&arguments[1]),
                            None,
                            None,
                            Some(&arguments[2]),
                        ),
                        "prob(of:given:prior:confidence:)" => (
                            &arguments[0],
                            Some(&arguments[1]),
                            Some(&arguments[2]),
                            Some(&arguments[3]),
                            None,
                        ),
                        "prob(of:given:prior:confidence:over:)" => (
                            &arguments[0],
                            Some(&arguments[1]),
                            Some(&arguments[2]),
                            Some(&arguments[3]),
                            Some(&arguments[4]),
                        ),
                        _ => return Ok(ChangeSet::empty()),
                    };
                Self::desugar_prob(
                    expr,
                    ast,
                    origin,
                    stream,
                    of_expr,
                    given_expr,
                    prior_expr,
                    confidence_expr,
                    over_expr,
                )
            }
            _ => Ok(ChangeSet::empty()),
        }
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
