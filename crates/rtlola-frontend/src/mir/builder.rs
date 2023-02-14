use rtlola_hir::hir::Output;
use rtlola_reporting::{Diagnostic, RtLolaError};

use crate::mir::{
    DiscreteWindow, EventDrivenStream, Expression, ExpressionKind, InputStream, OutputStream, SlidingWindow,
    TimeDrivenStream, Trigger, Type, Window,
};
use crate::RtLolaMir;

trait Feature {
    fn name(&self) -> &'static str;

    fn exclude_input(&self, input: &InputStream) -> bool {
        false
    }

    fn exclude_output(&self, output: &OutputStream) -> bool {
        false
    }

    fn exclude_time_driven(&self, stream: &TimeDrivenStream) -> bool {
        false
    }

    fn exclude_event_driven(&self, stream: &EventDrivenStream) -> bool {
        false
    }

    fn exclude_discrete_window(&self, window: &DiscreteWindow) -> bool {
        false
    }

    fn exclude_sliding_window(&self, window: &SlidingWindow) -> bool {
        false
    }

    fn exclude_trigger(&self, trigger: &Trigger) -> bool {
        false
    }

    fn exclude_type(&self, ty: &Type) -> bool {
        false
    }

    fn exclude_expression_kind(&self, kind: &ExpressionKind) -> bool {
        false
    }
}

#[allow(missing_debug_implementations)]
pub struct MirBuilder {
    mir: RtLolaMir,
    features: Vec<Box<dyn Feature>>,
}

impl Feature for MirBuilder {
    fn name(&self) -> &'static str {
        "MirBuilder"
    }

    fn exclude_input(&self, input: &InputStream) -> bool {
        self.features.iter().any(|f| f.exclude_input(input))
    }

    fn exclude_output(&self, output: &OutputStream) -> bool {
        self.features.iter().any(|f| f.exclude_output(output))
    }

    fn exclude_time_driven(&self, stream: &TimeDrivenStream) -> bool {
        self.features.iter().any(|f| f.exclude_time_driven(stream))
    }

    fn exclude_event_driven(&self, stream: &EventDrivenStream) -> bool {
        self.features.iter().any(|f| f.exclude_event_driven(stream))
    }

    fn exclude_discrete_window(&self, window: &DiscreteWindow) -> bool {
        self.features.iter().any(|f| f.exclude_discrete_window(window))
    }

    fn exclude_sliding_window(&self, window: &SlidingWindow) -> bool {
        self.features.iter().any(|f| f.exclude_sliding_window(window))
    }

    fn exclude_trigger(&self, trigger: &Trigger) -> bool {
        self.features.iter().any(|f| f.exclude_trigger(trigger))
    }

    fn exclude_type(&self, ty: &Type) -> bool {
        self.features.iter().any(|f| f.exclude_type(ty))
            || match ty {
                Type::Bool | Type::Int(_) | Type::UInt(_) | Type::Float(_) | Type::String | Type::Bytes => false, /* handled by first disjunct */
                Type::Tuple(children) => children.iter().any(|ty| self.exclude_type(ty)),
                Type::Option(ty) => self.exclude_type(ty.as_ref()),
                Type::Function { args, ret } => {
                    args.iter().any(|ty| self.exclude_type(ty)) || self.exclude_type(ret.as_ref())
                },
            }
    }

    fn exclude_expression_kind(&self, kind: &ExpressionKind) -> bool {
        self.features.iter().any(|f| f.exclude_expression_kind(kind))
    }
}

impl MirBuilder {
    pub fn new(mir: RtLolaMir) -> Self {
        Self { mir, features: vec![] }
    }

    pub fn build(self) -> Result<RtLolaMir, RtLolaError> {
        let res = self
            .mir
            .inputs
            .iter()
            .any(|i| self.exclude_input(i) || self.exclude_type(&i.ty))
            || self.mir.outputs.iter().any(|o| {
                self.exclude_output(o)
                    || self.exclude_type(&o.ty)
                    || o.spawn
                        .condition
                        .as_ref()
                        .map(|exp| self.exclude_expression(&exp))
                        .unwrap_or(false)
                    || o.spawn
                        .expression
                        .as_ref()
                        .map(|exp| self.exclude_expression(&exp))
                        .unwrap_or(false)
                    || self.exclude_expression(&o.eval.expression)
                    || o.eval
                        .condition
                        .as_ref()
                        .map(|exp| self.exclude_expression(&exp))
                        .unwrap_or(false)
                    || o.close
                        .condition
                        .as_ref()
                        .map(|exp| self.exclude_expression(&exp))
                        .unwrap_or(false)
            })
            || self.mir.time_driven.iter().any(|td| self.exclude_time_driven(td))
            || self.mir.event_driven.iter().any(|ed| self.exclude_event_driven(ed))
            || self
                .mir
                .discrete_windows
                .iter()
                .any(|window| self.exclude_discrete_window(window) || self.exclude_type(&window.ty))
            || self
                .mir
                .sliding_windows
                .iter()
                .any(|window| self.exclude_sliding_window(window) || self.exclude_type(&window.ty))
            || self.mir.triggers.iter().any(|t| self.exclude_trigger(t));

        if res {
            let mut err = RtLolaError::new();
            err.add(Diagnostic::error(
                "The specification contains features unsupported by the backend",
            ));
            Err(err)
        } else {
            Ok(self.mir)
        }
    }

    // Recursively walk expression ast.
    fn exclude_expression(&self, exp: &Expression) -> bool {
        self.exclude_type(&exp.ty)
            || self.exclude_expression_kind(&exp.kind)
            || match &exp.kind {
                ExpressionKind::ParameterAccess(_, _) | ExpressionKind::LoadConstant(_) => false,
                ExpressionKind::Function(_, sub_exps)
                | ExpressionKind::Tuple(sub_exps)
                | ExpressionKind::StreamAccess {
                    target: _,
                    parameters: sub_exps,
                    access_kind: _,
                }
                | ExpressionKind::ArithLog(_, sub_exps) => sub_exps.iter().any(|exp| self.exclude_expression(exp)),
                ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                } => {
                    self.exclude_expression(condition.as_ref())
                        || self.exclude_expression(consequence.as_ref())
                        || self.exclude_expression(alternative.as_ref())
                },
                ExpressionKind::TupleAccess(target, _) => self.exclude_expression(target.as_ref()),
                ExpressionKind::Convert { expr } => self.exclude_expression(expr.as_ref()),
                ExpressionKind::Default { expr, default } => {
                    self.exclude_expression(expr.as_ref()) || self.exclude_expression(default.as_ref())
                },
            }
    }
}
