use std::rc::Rc;

use rtlola_reporting::{Diagnostic, RtLolaError};

use crate::{
    ast::{Expression, ExpressionKind, InstanceOperation, InstanceSelection, LambdaExpr, Literal},
    syntactic_sugar::builder::Builder,
    RtLolaAst,
};

use super::{ChangeSet, ExprOrigin, SynSugar};

pub(crate) struct ProbabilityAggregation;

impl ProbabilityAggregation {
    fn apply(&self, expr: &Expression, ast: &RtLolaAst) -> Result<ChangeSet, RtLolaError> {
        let ExpressionKind::Method(stream, name, _types, arguments) = &expr.kind else {
            return Ok(ChangeSet::empty());
        };
        if name.name.name != "prob" {
            return Ok(ChangeSet::empty());
        }

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
            name => {
                return Err(Diagnostic::error(&format!(
                    "Unsupported arguments to \"prob\" function: {name}",
                ))
                .add_span_with_label(expr.span, Some("Found prob method here"), true)
                .into())
            }
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

        let both_aggregation_cond = if let Some(given) = given_expr {
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

            builder.and(of_expr.next_id(ast), given_expr.next_id(ast))
        } else {
            of_expr.next_id(ast)
        };

        let both_aggregation = builder.instance_aggregation(
            (**stream).to_owned(),
            InstanceSelection::FilteredAll(LambdaExpr {
                parameters: of_parameters
                    .iter()
                    .map(|p| Rc::new(p.next_id(ast)))
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
                (**stream).to_owned(),
                InstanceSelection::FilteredAll(given_lambda.next_id(ast)),
                InstanceOperation::Count,
            )
        } else {
            builder.instance_aggregation(
                (**stream).to_owned(),
                InstanceSelection::All,
                InstanceOperation::Count,
            )
        };

        let denom = if let Some(confidence) = confidence_expr {
            builder.parentesized(builder.add(given_aggregation, confidence.next_id(ast)))
        } else {
            given_aggregation
        };

        let numerator = if let Some(prior) = prior_expr {
            builder.parentesized(builder.add(
                both_aggregation,
                builder.mul(prior.next_id(ast), confidence_expr.unwrap().next_id(ast)),
            ))
        } else {
            both_aggregation
        };

        let const_0 = builder.literal(Literal::new_numeric(
            ast.next_id(),
            "0.0",
            None,
            builder.span.to_indirect(),
        ));

        let new_expr = builder.div(numerator, denom.next_id(ast));

        let new_expr = builder.if_then_else(
            builder.eq(denom.next_id(ast), const_0.next_id(ast)),
            const_0.next_id(ast),
            new_expr,
        );

        Ok(ChangeSet::replace_current_expression(new_expr))
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
