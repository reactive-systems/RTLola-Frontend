use crate::hir::modes::memory_bounds::MemoryAnalyzed;
use crate::hir::modes::ordering::EvaluationOrderBuilt;
use crate::hir::modes::types::HirType;
use crate::hir::modes::types::TypeChecked;
use crate::hir::modes::Complete;
use crate::hir::StreamReference;
use crate::{hir, hir::Hir, mir, mir::Mir};

use super::{dependencies::WithDependencies, ir_expr::WithIrExpr};

impl Hir<Complete> {
    pub(crate) fn lower(self) -> Mir {
        let Hir { inputs, outputs, triggers, mode, .. } = self.clone();
        let inputs = inputs
            .into_iter()
            .map(|i| {
                let sr = i.sr;
                mir::InputStream {
                    name: i.name,
                    ty: Self::lower_type(self.stream_type(sr)),
                    acccessed_by: mode.direct_accesses(sr),
                    aggregates: mode.aggregates(sr),
                    layer: mode.stream_layers(sr),
                    memory_bound: mode.memory_bound(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::InputStream>>();
        let outputs = outputs
            .into_iter()
            .map(|o| {
                let sr = o.sr;
                mir::OutputStream {
                    name: o.name,
                    ty: Self::lower_type(self.stream_type(sr)),
                    expr: self.lower_expr(self.expr(sr)),
                    acccesses: mode.direct_accesses(sr),
                    acccessed_by: mode.direct_accessed_by(sr),
                    aggregates: mode.aggregates(sr),
                    memory_bound: mode.memory_bound(sr),
                    layer: mode.stream_layers(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::OutputStream>>();
        let time_driven = outputs
            .iter()
            .filter(|o| mode.is_periodic(o.reference))
            .map(|o| Self::lower_periodic(o.reference))
            .collect::<Vec<mir::TimeDrivenStream>>();
        let event_driven = outputs
            .iter()
            .filter(|o| mode.is_event(o.reference))
            .map(|o| mir::EventDrivenStream { reference: o.reference })
            .collect::<Vec<mir::EventDrivenStream>>();

        let (sliding_windows, discrete_windows) = mode.all_windows();
        let discrete_windows = discrete_windows
            .into_iter()
            .map(|win| self.lower_discrete_window(win))
            .collect::<Vec<mir::DiscreteWindow>>();
        let sliding_windows =
            sliding_windows.into_iter().map(|win| self.lower_sliding_window(win)).collect::<Vec<mir::SlidingWindow>>();
        let triggers = triggers
            .into_iter()
            .map(|t| mir::Trigger { message: t.message, reference: t.sr })
            .collect::<Vec<mir::Trigger>>();
        Mir { inputs, outputs, triggers, event_driven, time_driven, sliding_windows, discrete_windows }
    }

    fn lower_periodic(_sr: StreamReference) -> mir::TimeDrivenStream {
        todo!()
    }

    fn lower_sliding_window(&self, win: hir::SlidingWindow) -> mir::SlidingWindow {
        mir::SlidingWindow {
            target: win.target,
            caller: win.caller,
            duration: win.duration,
            wait: win.wait,
            op: win.op,
            reference: win.reference,
            ty: Self::lower_type(self.expr_type(win.eid)),
        }
    }

    fn lower_discrete_window(&self, win: hir::DiscreteWindow) -> mir::DiscreteWindow {
        mir::DiscreteWindow {
            target: win.target,
            caller: win.caller,
            duration: win.duration,
            wait: win.wait,
            op: win.op,
            reference: win.reference,
            ty: Self::lower_type(self.expr_type(win.eid)),
        }
    }

    fn lower_type(_ty: HirType) -> mir::Type {
        todo!("Implement me, when HIRTypes are fixed")
    }

    fn lower_expr(&self, expr: &hir::expression::Expression) -> mir::Expression {
        mir::Expression {
            kind: self.lower_expression_kind(&expr.kind),
            ty: Self::lower_type(self.mode.expr_type(expr.eid)),
        }
    }

    fn lower_expression_kind(&self, expr: &hir::expression::ExpressionKind) -> mir::ExpressionKind {
        match expr {
            hir::expression::ExpressionKind::LoadConstant(constant) => {
                mir::ExpressionKind::LoadConstant(Self::lower_constant(constant))
            }
            hir::expression::ExpressionKind::ArithLog(op, args) => {
                let op = Self::lower_arith_log_op(*op);
                let args = args.iter().map(|arg| self.lower_expr(arg)).collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::ArithLog(op, args)
            }
            hir::expression::ExpressionKind::StreamAccess(sr, kind, para) => {
                assert!(para.is_empty(), "Parametrization currently not implemented");
                mir::ExpressionKind::StreamAccess(*sr, *kind, para.iter().map(|p| self.lower_expr(p)).collect())
            }
            hir::expression::ExpressionKind::ParameterAccess(_sr, _para) => unimplemented!(),
            hir::expression::ExpressionKind::Ite { condition, consequence, alternative } => {
                let condition = Box::new(self.lower_expr(condition));
                let consequence = Box::new(self.lower_expr(consequence));
                let alternative = Box::new(self.lower_expr(alternative));
                mir::ExpressionKind::Ite { condition, consequence, alternative }
            }
            hir::expression::ExpressionKind::Tuple(elements) => {
                let elements =
                    elements.iter().map(|element| self.lower_expr(element)).collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Tuple(elements)
            }
            hir::expression::ExpressionKind::TupleAccess(tuple, element_pos) => {
                let tuple = Box::new(self.lower_expr(tuple));
                let element_pos = *element_pos;
                mir::ExpressionKind::TupleAccess(tuple, element_pos)
            }
            hir::expression::ExpressionKind::Function { name, args, type_param: _ } => {
                let args = args.iter().map(|arg| self.lower_expr(arg)).collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Function(name.clone(), args)
            }
            hir::expression::ExpressionKind::Widen(expr, _ty) => {
                let expr = Box::new(self.lower_expr(expr));
                mir::ExpressionKind::Convert { expr }
            }
            hir::expression::ExpressionKind::Default { expr, default } => {
                let expr = Box::new(self.lower_expr(expr));
                let default = Box::new(self.lower_expr(default));
                mir::ExpressionKind::Default { expr, default }
            }
        }
    }

    fn lower_constant(constant: &hir::expression::Constant) -> mir::Constant {
        match constant {
            hir::expression::Constant::BasicConstant(lit) => Self::lower_constant_literal(lit),
            hir::expression::Constant::InlinedConstant(lit, _ty) => Self::lower_constant_literal(lit),
        }
    }

    fn lower_constant_literal(constant: &hir::expression::ConstantLiteral) -> mir::Constant {
        match constant {
            hir::expression::ConstantLiteral::Str(s) => mir::Constant::Str(s.clone()),
            hir::expression::ConstantLiteral::Bool(b) => mir::Constant::Bool(*b),
            hir::expression::ConstantLiteral::Integer(_i) => todo!("type information needed"),
            hir::expression::ConstantLiteral::SInt(_i) => todo!("review needed"),
            hir::expression::ConstantLiteral::Float(f) => mir::Constant::Float(*f),
        }
    }

    fn lower_arith_log_op(op: hir::expression::ArithLogOp) -> mir::ArithLogOp {
        match op {
            hir::expression::ArithLogOp::Not => mir::ArithLogOp::Not,
            hir::expression::ArithLogOp::Neg => mir::ArithLogOp::Neg,
            hir::expression::ArithLogOp::Add => mir::ArithLogOp::Add,
            hir::expression::ArithLogOp::Sub => mir::ArithLogOp::Sub,
            hir::expression::ArithLogOp::Mul => mir::ArithLogOp::Mul,
            hir::expression::ArithLogOp::Div => mir::ArithLogOp::Div,
            hir::expression::ArithLogOp::Rem => mir::ArithLogOp::Rem,
            hir::expression::ArithLogOp::Pow => mir::ArithLogOp::Pow,
            hir::expression::ArithLogOp::And => mir::ArithLogOp::And,
            hir::expression::ArithLogOp::Or => mir::ArithLogOp::Or,
            hir::expression::ArithLogOp::BitXor => mir::ArithLogOp::BitXor,
            hir::expression::ArithLogOp::BitAnd => mir::ArithLogOp::BitAnd,
            hir::expression::ArithLogOp::BitOr => mir::ArithLogOp::BitOr,
            hir::expression::ArithLogOp::BitNot => mir::ArithLogOp::BitNot,
            hir::expression::ArithLogOp::Shl => mir::ArithLogOp::Shr,
            hir::expression::ArithLogOp::Shr => mir::ArithLogOp::Shr,
            hir::expression::ArithLogOp::Eq => mir::ArithLogOp::Eq,
            hir::expression::ArithLogOp::Lt => mir::ArithLogOp::Lt,
            hir::expression::ArithLogOp::Le => mir::ArithLogOp::Le,
            hir::expression::ArithLogOp::Ne => mir::ArithLogOp::Ne,
            hir::expression::ArithLogOp::Ge => mir::ArithLogOp::Ge,
            hir::expression::ArithLogOp::Gt => mir::ArithLogOp::Gt,
        }
    }
}
