use rtlola_reporting::{Diagnostic, RtLolaError, Span};

use crate::hir::{
    AnnotatedPacingType, ConcretePacingType, ConcreteValueType, DepAnaTrait, DiscreteAggr, Expression, ExpressionKind,
    FnExprKind, Input, MemBoundTrait, OrderedTrait, Output, SlidingAggr, StreamType, Trigger, TypedTrait,
    WidenExprKind, Window,
};
use crate::{CompleteMode, RtLolaHir};

trait Feature {
    fn name(&self) -> &'static str;

    fn exclude_input(&self, input: &Input) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_discrete_window(&self, window: &Window<DiscreteAggr>) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_sliding_window(&self, window: &Window<SlidingAggr>) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_trigger(&self, trigger: &Trigger) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_value_type(&self, span: &Span, ty: &ConcreteValueType) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_pacing_type(&self, span: &Span, ty: &ConcretePacingType) -> Result<(), RtLolaError> {
        Ok(())
    }

    fn exclude_expression_kind(&self, span: &Span, kind: &ExpressionKind) -> Result<(), RtLolaError> {
        Ok(())
    }
}

#[allow(missing_debug_implementations)]
pub struct FeatureValidator {
    hir: RtLolaHir<CompleteMode>,
    features: Vec<Box<dyn Feature>>,
}

impl Feature for FeatureValidator {
    fn name(&self) -> &'static str {
        "FeatureValidator"
    }

    fn exclude_input(&self, input: &Input) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_input(input))
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_output(output))
    }

    fn exclude_discrete_window(&self, window: &Window<DiscreteAggr>) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_discrete_window(window))
    }

    fn exclude_sliding_window(&self, window: &Window<SlidingAggr>) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_sliding_window(window))
    }

    fn exclude_trigger(&self, trigger: &Trigger) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_trigger(trigger))
    }

    fn exclude_value_type(&self, span: &Span, ty: &ConcreteValueType) -> Result<(), RtLolaError> {
        let current = self.iter_features(|f| f.exclude_value_type(span, ty));
        let other = match ty {
            ConcreteValueType::Bool
            | ConcreteValueType::Integer8
            | ConcreteValueType::Integer16
            | ConcreteValueType::Integer32
            | ConcreteValueType::Integer64
            | ConcreteValueType::UInteger8
            | ConcreteValueType::UInteger16
            | ConcreteValueType::UInteger32
            | ConcreteValueType::UInteger64
            | ConcreteValueType::Float32
            | ConcreteValueType::Float64
            | ConcreteValueType::TString
            | ConcreteValueType::Byte => Ok(()), /* handled by first disjunct */
            ConcreteValueType::Tuple(children) => {
                children
                    .iter()
                    .flat_map(|ty| self.exclude_value_type(span, ty).map_err(|e| e.into_iter()).err())
                    .flatten()
                    .collect::<RtLolaError>()
                    .into()
            },
            ConcreteValueType::Option(ty) => self.exclude_value_type(span, ty.as_ref()),
        };
        let mut res = RtLolaError::new();
        if let Err(e) = current {
            res.join(e);
        }
        if let Err(e) = other {
            res.join(e)
        }
        res.into()
    }

    fn exclude_pacing_type(&self, span: &Span, ty: &ConcretePacingType) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_pacing_type(span, ty))
    }

    fn exclude_expression_kind(&self, span: &Span, kind: &ExpressionKind) -> Result<(), RtLolaError> {
        self.iter_features(|f| f.exclude_expression_kind(span, kind))
    }
}

impl FeatureValidator {
    pub fn new(hir: RtLolaHir<CompleteMode>) -> Self {
        Self { hir, features: vec![] }
    }

    pub fn build(self) -> Result<RtLolaHir<CompleteMode>, RtLolaError> {
        let mut res = RtLolaError::new();
        self.hir.inputs().for_each(|i| {
            let ty = self.hir.stream_type(i.sr);
            if let Err(e) = self.exclude_input(i) {
                res.join(e);
            }
            if let Err(e) = self.exclude_value_type(&i.span, &ty.value_ty) {
                res.join(e);
            }
        });

        self.hir.outputs.iter().for_each(|o| {
            let ty = self.hir.stream_type(o.sr);
            let spawn_span = &o
                .spawn
                .as_ref()
                .and_then(|spawn| {
                    spawn
                        .expression
                        .map(|expr| self.hir.expression(expr).span.clone())
                        .or_else(|| spawn.condition.map(|expr| self.hir.expression(expr).span.clone()))
                        .or_else(|| {
                            spawn.pacing.as_ref().map(|apt| {
                                match &apt {
                                    AnnotatedPacingType::Frequency { span, .. } => span.clone(),
                                    AnnotatedPacingType::Expr(eid) => self.hir.expression(*eid).span.clone(),
                                }
                            })
                        })
                })
                .unwrap_or(Span::Unknown);

            if let Err(e) = self.exclude_output(o) {
                res.join(e);
            }
            if let Err(e) = self.exclude_value_type(&o.span, &ty.value_ty) {
                res.join(e);
            }
            if let Err(e) = self.exclude_pacing_type(&o.span, &ty.pacing_ty) {
                res.join(e);
            }
            if let Err(e) = self.exclude_pacing_type(spawn_span, &ty.spawn.0) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.spawn_expr(o.sr)) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.spawn_cond(o.sr)) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.eval_expr(o.sr)) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.eval_cond(o.sr)) {
                res.join(e);
            }
            if let Err(e) = self.exclude_expression_opt(self.hir.close_cond(o.sr)) {
                res.join(e);
            }
        });
        self.hir.discrete_windows().iter().for_each(|window| {
            if let Err(e) = self.exclude_discrete_window(window) {
                res.join(e);
            }
        });
        self.hir.sliding_windows().iter().for_each(|window| {
            if let Err(e) = self.exclude_sliding_window(window) {
                res.join(e);
            }
        });
        self.hir.triggers.iter().for_each(|t| {
            if let Err(e) = self.exclude_trigger(t) {
                res.join(e);
            }
        });

        Result::from(res).map(|_| self.hir)
    }

    fn iter_features<F>(&self, op: F) -> Result<(), RtLolaError>
    where
        F: Fn(&Box<dyn Feature>) -> Result<(), RtLolaError>,
    {
        self.features
            .iter()
            .flat_map(|f| op(f).map_err(|e| e.into_iter()).err())
            .flatten()
            .collect::<RtLolaError>()
            .into()
    }

    fn exclude_expression_opt(&self, exp: Option<&Expression>) -> Result<(), RtLolaError> {
        exp.map(|e| self.exclude_expression(e)).unwrap_or(Ok(()))
    }

    // Recursively walk expression ast.
    fn exclude_expression(&self, exp: &Expression) -> Result<(), RtLolaError> {
        let span = &exp.span;
        let stream_ty: StreamType = self.hir.expr_type(exp.eid);
        let mut res = RtLolaError::new();
        if let Err(e) = self.exclude_value_type(span, &stream_ty.value_ty) {
            res.join(e);
        }
        if let Err(e) = self.exclude_expression_kind(span, &exp.kind) {
            res.join(e);
        }
        match &exp.kind {
            ExpressionKind::ParameterAccess(_, _) | ExpressionKind::LoadConstant(_) => {},
            ExpressionKind::Function(FnExprKind {
                name: _,
                args: sub_exps,
                type_param: _,
            })
            | ExpressionKind::Tuple(sub_exps)
            | ExpressionKind::StreamAccess(_, _, sub_exps)
            | ExpressionKind::ArithLog(_, sub_exps) => {
                sub_exps.iter().for_each(|exp| {
                    if let Err(e) = self.exclude_expression(exp) {
                        res.join(e)
                    }
                })
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                if let Err(e) = self.exclude_expression(condition.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(consequence.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(alternative.as_ref()) {
                    res.join(e)
                }
            },
            ExpressionKind::TupleAccess(target, _) => {
                if let Err(e) = self.exclude_expression(target.as_ref()) {
                    res.join(e);
                }
            },
            ExpressionKind::Widen(WidenExprKind { expr, ty: _ }) => {
                if let Err(e) = self.exclude_expression(expr.as_ref()) {
                    res.join(e);
                }
            },
            ExpressionKind::Default { expr, default } => {
                if let Err(e) = self.exclude_expression(expr.as_ref()) {
                    res.join(e);
                }
                if let Err(e) = self.exclude_expression(default.as_ref()) {
                    res.join(e);
                }
            },
        };

        res.into()
    }
}
