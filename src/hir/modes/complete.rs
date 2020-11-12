use crate::common_ir::SRef;
use crate::hir::modes::dependencies::{DependenciesWrapper, WithDependencies};

use crate::hir::modes::ir_expr::IrExprWrapper;
use crate::hir::modes::memory_bounds::{MemoryAnalyzed, MemoryWrapper};
use crate::hir::modes::ordering::OrderedWrapper;
use crate::hir::modes::types::{HirType, TypeChecked, TypedWrapper};
use crate::hir::modes::Complete;
use crate::hir::modes::Dependencies;
use crate::hir::modes::EvaluationOrder;
use crate::hir::modes::IrExpression;
use crate::hir::modes::Memory;
use crate::hir::modes::Typed;
use crate::hir::StreamReference;
use crate::{common_ir::Tracking, hir::Window};
use crate::{hir, hir::Hir, mir, mir::Mir};

impl Hir<Complete> {
    #[allow(unreachable_code)]
    pub(crate) fn lower(self) -> Mir {
        let Hir { inputs, outputs, triggers, mode, .. } = self.clone();
        let inputs = inputs
            .into_iter()
            .map(|i| {
                let sr = i.sr;
                mir::InputStream {
                    name: i.name,
                    ty: Self::lower_type(self.stream_type(sr)),
                    // dependent_streams: mode.accessed_by(sr).into_iter().map(|sr| Self::lower_dependency(*sr)).collect(),
                    dependent_streams: mode.accessed_by(sr).to_vec(),
                    dependent_windows: mode.aggregated_by(sr).to_vec(),
                    layer: todo!("Fix type error"), //mode.layers(sr),
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
                    input_dependencies: mode.accesses(sr).into_iter().filter(SRef::is_input).collect(), // TODO: Is this supposed to be transitive?
                    outgoing_dependencies: mode.accesses(sr).into_iter().filter(|_sr| todo!()).collect(), // TODO: Is this supposed to be transitive?
                    dependent_streams: mode.accessed_by(sr).into_iter().map(Self::lower_dependency).collect(),
                    dependent_windows: mode.aggregated_by(sr).into_iter().map(|(_sr, wr)| wr).collect(),
                    memory_bound: mode.memory_bound(sr),
                    layer: todo!("Fix type error"), //self.layers(sr),
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

        let sliding_windows = self.windows().into_iter().map(Self::lower_window).collect::<Vec<mir::SlidingWindow>>();
        let triggers = triggers
            .into_iter()
            .map(|t| mir::Trigger { message: t.message, reference: t.sr })
            .collect::<Vec<mir::Trigger>>();
        Mir { inputs, outputs, triggers, event_driven, time_driven, sliding_windows, discrete_windows: todo!() }
    }

    fn lower_periodic(_sr: StreamReference) -> mir::TimeDrivenStream {
        todo!("")
    }

    fn lower_window(_win: Window) -> mir::SlidingWindow {
        todo!()
    }

    fn lower_type(_ty: HirType) -> mir::Type {
        todo!("Implement me, when HIRTypes are fixed")
    }

    fn lower_dependency(_dep: StreamReference) -> Tracking {
        todo!()
    }

    fn lower_expr(&self, expr: &hir::expression::Expression) -> mir::Expression {
        mir::Expression {
            kind: self.lower_expression_kind(&expr.kind),
            ty: Self::lower_type(self.mode.expr_type(expr.eid)),
        }
    }

    #[allow(unreachable_code, unused_variables)]
    fn lower_expression_kind(&self, expr: &hir::expression::ExpressionKind) -> mir::ExpressionKind {
        match expr {
            hir::expression::ExpressionKind::LoadConstant(constant) => {
                mir::ExpressionKind::LoadConstant(Self::lower_constant(constant))
            }
            hir::expression::ExpressionKind::ArithLog(op, args) => {
                let op = Self::lower_arith_log_op(*op);
                let args = args.iter().map(|arg| self.lower_expr(arg)).collect::<Vec<mir::Expression>>();
                let ty = todo!();
                mir::ExpressionKind::ArithLog(op, args, ty)
            }
            hir::expression::ExpressionKind::StreamAccess(sr, kind, para) => {
                assert!(para.is_empty(), "Parametrization currently not implemented");
                match kind {
                    // TODO: Change StreamAccesses in MIR
                    hir::expression::StreamAccessKind::Sync => {
                        mir::ExpressionKind::StreamAccess(*sr, mir::StreamAccessKind::Sync)
                    }
                    hir::expression::StreamAccessKind::Hold => {
                        mir::ExpressionKind::StreamAccess(*sr, mir::StreamAccessKind::Hold)
                    }
                    hir::expression::StreamAccessKind::Offset(off) => {
                        mir::ExpressionKind::OffsetLookup { target: *sr, offset: *off }
                    }
                    hir::expression::StreamAccessKind::DiscreteWindow(wr) => mir::ExpressionKind::WindowLookup(*wr),
                    hir::expression::StreamAccessKind::SlidingWindow(wr) => mir::ExpressionKind::WindowLookup(*wr),
                }
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
                let return_ty = todo!();
                mir::ExpressionKind::Function(name.clone(), args, return_ty)
            }
            hir::expression::ExpressionKind::Widen(expr, ty) => {
                let from = todo!();
                let to = todo!();
                let expr = Box::new(self.lower_expr(expr));
                mir::ExpressionKind::Convert { from, to, expr }
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

impl IrExprWrapper for Complete {
    type InnerE = IrExpression;
    fn inner_expr(&self) -> &Self::InnerE {
        &self.ir_expr
    }
}

impl DependenciesWrapper for Complete {
    type InnerD = Dependencies;
    fn inner_dep(&self) -> &Self::InnerD {
        &self.dependencies
    }
}

impl MemoryWrapper for Complete {
    type InnerM = Memory;
    fn inner_memory(&self) -> &Self::InnerM {
        &self.memory
    }
}

impl TypedWrapper for Complete {
    type InnerT = Typed;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.types
    }
}

impl OrderedWrapper for Complete {
    type InnerO = EvaluationOrder;
    fn inner_order(&self) -> &Self::InnerO {
        &self.layers
    }
}
