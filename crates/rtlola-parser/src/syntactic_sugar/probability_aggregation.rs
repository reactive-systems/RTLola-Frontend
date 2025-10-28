use std::rc::Rc;

use rtlola_reporting::{Diagnostic, RtLolaError};

use crate::{
    ast::{
        Expression, ExpressionKind, InstanceOperation, InstanceSelection, LambdaExpr, Parameter,
    },
    syntactic_sugar::builder::Builder,
    RtLolaAst,
};

use super::{ChangeSet, ExprOrigin, SynSugar};

pub(crate) struct ProbabilityAggregation;

impl ProbabilityAggregation {
    #[cfg(not(feature = "probability"))]
    fn desugar_instance_prob(
        builder: Builder,
        stream: &Expression,
        parameters: &Vec<Rc<Parameter>>,
        of_expr: &Expression,
        given_expr: Option<&Expression>,
        confidence_expr: Option<&Expression>,
        prior_expr: Option<&Expression>,
    ) -> Result<ChangeSet, RtLolaError> {
        let both_aggregation_cond = if let Some(given_expr) = given_expr {
            builder.and(
                of_expr.next_id(builder.ast),
                given_expr.next_id(builder.ast),
            )
        } else {
            of_expr.next_id(builder.ast)
        };

        let both_aggregation = builder.instance_aggregation(
            stream.to_owned(),
            InstanceSelection::FilteredAll(LambdaExpr {
                parameters: parameters
                    .iter()
                    .map(|p| Rc::new(p.next_id(builder.ast)))
                    .collect(),
                expr: Box::new(both_aggregation_cond),
            }),
            InstanceOperation::Count,
        );

        let given_aggregation = if let Some(given) = given_expr {
            let ExpressionKind::Lambda(given_lambda) = &given.kind else {
                unreachable!()
            };

            builder.instance_aggregation(
                stream.to_owned(),
                InstanceSelection::FilteredAll(given_lambda.next_id(builder.ast)),
                InstanceOperation::Count,
            )
        } else {
            builder.instance_aggregation(
                stream.to_owned(),
                InstanceSelection::All,
                InstanceOperation::Count,
            )
        };

        let denom = if let Some(confidence) = confidence_expr {
            builder.parentesized(builder.add(given_aggregation, confidence.next_id(builder.ast)))
        } else {
            given_aggregation
        };

        let numerator = if let Some(prior) = prior_expr {
            builder.parentesized(builder.add(
                both_aggregation,
                builder.mul(
                    prior.next_id(builder.ast),
                    confidence_expr.unwrap().next_id(builder.ast),
                ),
            ))
        } else {
            both_aggregation
        };

        let const_0 = builder.literal(Literal::new_numeric(
            builder.ast.next_id(),
            "0.0",
            None,
            builder.span.to_indirect(),
        ));

        let new_expr = builder.div(numerator, denom.next_id(builder.ast));

        let new_expr = builder.if_then_else(
            builder.eq(denom.next_id(builder.ast), const_0.next_id(builder.ast)),
            const_0.next_id(builder.ast),
            new_expr,
        );

        Ok(ChangeSet::replace_current_expression(new_expr))
    }

    #[cfg(feature = "probability")]
    fn desugar_instance_prob(
        builder: Builder,
        target: usize,
        parameters: &Vec<Rc<Parameter>>,
        of_expr: &Expression,
        given_expr: Option<&Expression>,
        confidence_expr: Option<&Expression>,
        prior_expr: Option<&Expression>,
    ) -> Result<ChangeSet, RtLolaError> {
        use crate::ast::{AnnotatedPacingType, Ident, Output};

        let output = &builder.ast.outputs[target];
        assert!(!output.params.is_empty(), "target must be parameterized");
        for (output_param, lambda_param) in parameters.iter().zip(&output.params) {
            assert!(
                output_param.name.name == lambda_param.name.name,
                "lambda parameters must be named same as target"
            );
        }
        let target_stream_name = builder.ast.primed_name("target");
        let target_stream_ident = Ident::new(target_stream_name, builder.span.to_indirect());

        let target_expr = match (given_expr, confidence_expr, prior_expr) {
            (None, None, None) => of_expr.next_id(builder.ast),
            (Some(given_expr), None, None) => {
                builder.tuple(vec![of_expr.clone(), given_expr.clone()])
            }
            (Some(given_expr), Some(confidence_expr), Some(prior_expr)) => builder.tuple(vec![
                of_expr.clone(),
                given_expr.clone(),
                confidence_expr.clone(),
                prior_expr.clone(),
            ]),
            _ => unreachable!(),
        };

        let condition = if output.eval.len() == 1 {
            output.eval[0]
                .condition
                .as_ref()
                .map(|c| c.next_id(builder.ast))
        } else {
            None
        };

        let target_eval_spec = builder.eval_spec(
            condition,
            AnnotatedPacingType::NotAnnotated(builder.span.to_indirect()),
            Some(target_expr),
        );

        let target_stream = Output {
            kind: crate::ast::OutputKind::NamedOutput(target_stream_ident.next_id(builder.ast)),
            annotated_type: None,
            params: output
                .params
                .iter()
                .map(|p| Rc::new(p.next_id(builder.ast)))
                .collect(),
            spawn: output.spawn.as_ref().map(|s| s.next_id(builder.ast)),
            eval: vec![target_eval_spec],
            close: output.close.as_ref().map(|c| c.next_id(builder.ast)),
            tags: Vec::new(),
            id: builder.ast.next_id(),
            span: builder.span.to_indirect(),
        };

        let op = match (given_expr, prior_expr, confidence_expr) {
            (None, None, None) => InstanceOperation::TrueRatio,
            (Some(_), None, None) => InstanceOperation::ConditionalProbability,
            (Some(_), Some(_), Some(_)) => InstanceOperation::ConditionalProbabilityWithPrior,
            _ => unreachable!(),
        };

        let new_expr = builder.instance_aggregation(
            builder.ident(target_stream_ident),
            InstanceSelection::All,
            op,
        );

        Ok(ChangeSet::replace_current_expression(new_expr) + ChangeSet::add_output(target_stream))
    }

    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> Result<ChangeSet, RtLolaError> {
        let ExpressionKind::Method(stream, name, _types, arguments) = &expr.kind else {
            return Ok(ChangeSet::empty());
        };
        if name.name.name != "prob" {
            return Ok(ChangeSet::empty());
        }

        let ExpressionKind::Ident(stream_name) = &stream.kind else {
            panic!("prob with of: can only be used with parameterized stream method")
        };

        let Some(stream) = ast
            .outputs
            .iter()
            .position(|o| o.name().is_some_and(|name| name.name == stream_name.name))
        else {
            panic!("can find stream named \"{}\".", &stream_name.name)
        };

        let builder = Builder::new(expr.span, ast);

        let (of_expr, given_expr, prior_expr, confidence_expr) = match name.to_string().as_str() {
            "prob(of:)" => (&arguments[0], None, None, None),
            "prob(of:given:)" => (&arguments[0], Some(&arguments[1]), None, None),
            "prob(of:given:prior:confidence:)" => (
                &arguments[0],
                Some(&arguments[1]),
                Some(&arguments[2]),
                Some(&arguments[3]),
            ),
            _ => return Ok(ChangeSet::empty()),
        };

        let ExpressionKind::Lambda(LambdaExpr {
            parameters: of_parameters,
            expr: of_expr,
        }) = &of_expr.kind
        else {
            return Err(Diagnostic::error(
                "Prob method requires lambda function for the of-argument.",
            )
            .add_span_with_label(of_expr.span, Some("Found of-argument here."), true)
            .into());
        };

        let given_expr = if let Some(given) = given_expr {
            let ExpressionKind::Lambda(LambdaExpr {
                parameters: given_parameters,
                expr: given_expr,
            }) = &given.kind
            else {
                return Err(Diagnostic::error(
                    "Prob method requires lambda function for the given-argument.",
                )
                .add_span_with_label(given.span, Some("Found given-argument here."), true)
                .into());
            };

            if of_parameters.len() != given_parameters.len() {
                return Err(Diagnostic::error(
						"The lambda functions for of: and given: arguments of prob function require the same number of parameters."
					)
					.add_span_with_label(expr.span, Some("Found prob method here."), true)
					.into());
            }

            // otherwise it is a lot more difficult to merge the lambda's before naming analysis
            for (of_param, given_param) in of_parameters.iter().zip(given_parameters) {
                if of_param.name != given_param.name {
                    return Err(Diagnostic::error(
						"The lambda functions for of: and given: arguments of prob function require the same parameter names."
					)
					.add_span_with_label(expr.span, Some("Found prob method here."), true)
					.into());
                }
            }
            Some(&**given_expr)
        } else {
            None
        };

        Self::desugar_instance_prob(
            builder,
            stream,
            of_parameters,
            of_expr,
            given_expr,
            confidence_expr,
            prior_expr,
        )
    }
}

impl SynSugar for ProbabilityAggregation {
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
