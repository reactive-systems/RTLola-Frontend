use itertools::Itertools;
use rtlola_hir::hir::{
    ActivationCondition, ArithLogOp, ConcretePacingType, ConcreteValueType, Constant, DepAnaTrait, DiscreteAggr,
    Expression, ExpressionKind, FnExprKind, Inlined, MemBoundTrait, Offset, OrderedTrait, SlidingAggr,
    StreamAccessKind, StreamReference, TypedTrait, WidenExprKind, Window,
};
use rtlola_hir::{CompleteMode, RtLolaHir};
use rtlola_parser::ast::WindowOperation;

use crate::mir;
use crate::mir::{InstanceTemplate, Mir};

impl Mir {
    /// Generates an Mir from a complete Hir.
    pub(crate) fn from_hir(hir: RtLolaHir<CompleteMode>) -> Mir {
        let inputs = hir
            .inputs()
            .map(|i| {
                let sr = i.sr();
                mir::InputStream {
                    name: i.name.clone(),
                    ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                    accessed_by: hir.direct_accesses(sr),
                    aggregated_by: hir.aggregated_by(sr),
                    layer: hir.stream_layers(sr),
                    memory_bound: hir.memory_bound(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::InputStream>>();
        // assert that each sr is available
        let outputs = hir
            .outputs()
            .map(|o| {
                let sr = o.sr();
                mir::OutputStream {
                    name: o.name.clone(),
                    ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                    expr: Self::lower_expr(&hir, hir.expr(sr)),
                    instance_template: Self::lower_instance_template(&hir, sr),
                    accesses: hir.direct_accesses(sr),
                    accessed_by: hir.direct_accessed_by(sr),
                    aggregated_by: hir.aggregated_by(sr),
                    memory_bound: hir.memory_bound(sr),
                    layer: hir.stream_layers(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::OutputStream>>();
        let (trigger_streams, triggers): (Vec<mir::OutputStream>, Vec<mir::Trigger>) = hir
            .triggers()
            .sorted_by(|a, b| Ord::cmp(&a.sr(), &b.sr()))
            .enumerate()
            .map(|(index, t)| {
                let sr = t.sr();
                let mir_trigger = mir::Trigger {
                    message: t.message.clone(),
                    reference: sr,
                    trigger_reference: index,
                };
                let mir_output_stream = mir::OutputStream {
                    name: format!("trigger_{}", index), //TODO better name
                    ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                    expr: Self::lower_expr(&hir, hir.expr(sr)),
                    instance_template: None,
                    accesses: hir.direct_accesses(sr),
                    accessed_by: hir.direct_accessed_by(sr),
                    aggregated_by: hir.aggregated_by(sr),
                    memory_bound: hir.memory_bound(sr),
                    layer: hir.stream_layers(sr),
                    reference: sr,
                };
                (mir_output_stream, mir_trigger)
            })
            .unzip();
        let outputs = outputs
            .into_iter()
            .chain(trigger_streams.into_iter())
            .sorted_by(|a, b| Ord::cmp(&a.reference, &b.reference))
            .collect::<Vec<_>>();
        //TODO: change SR if streams are deleted during a transformation
        assert!(
            outputs
                .iter()
                .enumerate()
                .all(|(index, o)| index == o.reference.out_ix()),
            "SRefs need to enumerated from 0 to the number of streams"
        );
        let time_driven = outputs
            .iter()
            .filter(|o| hir.is_periodic(o.reference))
            .map(|o| Self::lower_periodic(&hir, o.reference))
            .collect::<Vec<mir::TimeDrivenStream>>();
        let event_driven = outputs
            .iter()
            .filter(|o| hir.is_event(o.reference))
            .map(|o| Self::lower_event_based(&hir, o.reference))
            .collect::<Vec<mir::EventDrivenStream>>();

        let discrete_windows = hir
            .discrete_windows()
            .into_iter()
            .map(|win| Self::lower_discrete_window(&hir, win))
            .collect::<Vec<mir::DiscreteWindow>>();
        let sliding_windows = hir
            .sliding_windows()
            .into_iter()
            .map(|win| Self::lower_sliding_window(&hir, win))
            .collect::<Vec<mir::SlidingWindow>>();
        Mir {
            inputs,
            outputs,
            triggers,
            event_driven,
            time_driven,
            sliding_windows,
            discrete_windows,
        }
    }

    fn lower_event_based(hir: &RtLolaHir<CompleteMode>, sr: StreamReference) -> mir::EventDrivenStream {
        if let ConcretePacingType::Event(ac) = hir.stream_type(sr).pacing_ty {
            mir::EventDrivenStream {
                reference: sr,
                ac: Self::lower_activation_condition(&ac),
            }
        } else {
            unreachable!()
        }
    }

    fn lower_activation_condition(ac: &ActivationCondition) -> mir::ActivationCondition {
        match ac {
            ActivationCondition::Conjunction(con) => {
                mir::ActivationCondition::Conjunction(con.iter().map(Self::lower_activation_condition).collect())
            },
            ActivationCondition::Disjunction(dis) => {
                mir::ActivationCondition::Disjunction(dis.iter().map(Self::lower_activation_condition).collect())
            },
            ActivationCondition::Stream(sr) => mir::ActivationCondition::Stream(*sr),
            ActivationCondition::True => mir::ActivationCondition::True,
        }
    }

    fn lower_periodic(hir: &RtLolaHir<CompleteMode>, sr: StreamReference) -> mir::TimeDrivenStream {
        if let ConcretePacingType::FixedPeriodic(freq) = &hir.stream_type(sr).pacing_ty {
            mir::TimeDrivenStream {
                reference: sr,
                frequency: *freq,
            }
        } else {
            unreachable!()
        }
    }

    fn lower_instance_template(_hir: &RtLolaHir<CompleteMode>, _sr: StreamReference) -> Option<InstanceTemplate> {
        // let HirType { value_ty: _, pacing_ty: _, spawn, filter, close } = self.stream_type(sr);
        // let spawn_cond = self.lower_expr(&spawn.1);
        // let filter = self.lower_expr(&filter);
        // let close = self.lower_expr(&close);
        // assert!(matches!(spawn.0, PacingTy::Constant));
        // assert_eq!(spawn_cond.kind, mir::ExpressionKind::LoadConstant(mir::Constant::Bool(true)));
        // assert_eq!(filter.kind, mir::ExpressionKind::LoadConstant(mir::Constant::Bool(true)));
        // assert_eq!(close.kind, mir::ExpressionKind::LoadConstant(mir::Constant::Bool(false)));
        None
    }

    fn lower_sliding_window(hir: &RtLolaHir<CompleteMode>, win: &Window<SlidingAggr>) -> mir::SlidingWindow {
        mir::SlidingWindow {
            target: win.target,
            caller: win.caller,
            duration: win.aggr.duration,
            wait: win.aggr.wait,
            op: Self::lower_window_operation(win.aggr.op),
            reference: win.reference(),
            ty: Self::lower_value_type(&hir.expr_type(win.id()).value_ty),
        }
    }

    fn lower_discrete_window(hir: &RtLolaHir<CompleteMode>, win: &Window<DiscreteAggr>) -> mir::DiscreteWindow {
        mir::DiscreteWindow {
            target: win.target,
            caller: win.caller,
            duration: win.aggr.duration,
            wait: win.aggr.wait,
            op: Self::lower_window_operation(win.aggr.op),
            reference: win.reference(),
            ty: Self::lower_value_type(&hir.expr_type(win.id()).value_ty),
        }
    }

    fn lower_value_type(ty: &ConcreteValueType) -> mir::Type {
        match ty {
            ConcreteValueType::Bool => mir::Type::Bool,
            ConcreteValueType::Integer8 => mir::Type::Int(mir::IntTy::Int8),
            ConcreteValueType::Integer16 => mir::Type::Int(mir::IntTy::Int16),
            ConcreteValueType::Integer32 => mir::Type::Int(mir::IntTy::Int32),
            ConcreteValueType::Integer64 => mir::Type::Int(mir::IntTy::Int64),
            ConcreteValueType::UInteger8 => mir::Type::UInt(mir::UIntTy::UInt8),
            ConcreteValueType::UInteger16 => mir::Type::UInt(mir::UIntTy::UInt16),
            ConcreteValueType::UInteger32 => mir::Type::UInt(mir::UIntTy::UInt32),
            ConcreteValueType::UInteger64 => mir::Type::UInt(mir::UIntTy::UInt64),
            ConcreteValueType::Float32 => mir::Type::Float(mir::FloatTy::Float32),
            ConcreteValueType::Float64 => mir::Type::Float(mir::FloatTy::Float64),
            ConcreteValueType::Tuple(elements) => {
                let elements = elements.iter().map(Self::lower_value_type).collect::<Vec<_>>();
                mir::Type::Tuple(elements)
            },
            ConcreteValueType::TString => mir::Type::String,
            ConcreteValueType::Byte => mir::Type::Bytes,
            ConcreteValueType::Option(v) => mir::Type::Option(Box::new(Self::lower_value_type(v))),
        }
    }

    fn lower_expr(hir: &RtLolaHir<CompleteMode>, expr: &Expression) -> mir::Expression {
        let ty = Self::lower_value_type(&hir.expr_type(expr.id()).value_ty);
        mir::Expression {
            kind: Self::lower_expression_kind(hir, &expr.kind, &ty),
            ty,
        }
    }

    fn lower_expression_kind(
        hir: &RtLolaHir<CompleteMode>,
        expr: &ExpressionKind,
        ty: &mir::Type,
    ) -> mir::ExpressionKind {
        match expr {
            rtlola_hir::hir::ExpressionKind::LoadConstant(constant) => {
                mir::ExpressionKind::LoadConstant(Self::lower_constant(constant, ty))
            },
            rtlola_hir::hir::ExpressionKind::ArithLog(op, args) => {
                let op = Self::lower_arith_log_op(*op);
                let args = args
                    .iter()
                    .map(|arg| Self::lower_expr(hir, arg))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::ArithLog(op, args)
            },
            rtlola_hir::hir::ExpressionKind::StreamAccess(sr, kind, para) => {
                mir::ExpressionKind::StreamAccess {
                    target: *sr,
                    access_kind: Self::lower_stream_access_kind(*kind),
                    parameters: para.iter().map(|p| Self::lower_expr(hir, p)).collect(),
                }
            },
            rtlola_hir::hir::ExpressionKind::ParameterAccess(_sr, _para) => unimplemented!(),
            rtlola_hir::hir::ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let condition = Box::new(Self::lower_expr(hir, condition));
                let consequence = Box::new(Self::lower_expr(hir, consequence));
                let alternative = Box::new(Self::lower_expr(hir, alternative));
                mir::ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                }
            },
            rtlola_hir::hir::ExpressionKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| Self::lower_expr(hir, element))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Tuple(elements)
            },
            rtlola_hir::hir::ExpressionKind::TupleAccess(tuple, element_pos) => {
                let tuple = Box::new(Self::lower_expr(hir, tuple));
                let element_pos = *element_pos;
                mir::ExpressionKind::TupleAccess(tuple, element_pos)
            },
            rtlola_hir::hir::ExpressionKind::Function(kind) => {
                let FnExprKind { name, args, .. } = kind;
                let args = args
                    .iter()
                    .map(|arg| Self::lower_expr(hir, arg))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Function(name.clone(), args)
            },
            rtlola_hir::hir::ExpressionKind::Widen(kind) => {
                let WidenExprKind { expr, .. } = kind;
                let expr = Box::new(Self::lower_expr(hir, expr));
                mir::ExpressionKind::Convert { expr }
            },
            rtlola_hir::hir::ExpressionKind::Default { expr, default } => {
                let expr = Box::new(Self::lower_expr(hir, expr));
                let default = Box::new(Self::lower_expr(hir, default));
                mir::ExpressionKind::Default { expr, default }
            },
        }
    }

    fn lower_constant(constant: &Constant, ty: &mir::Type) -> mir::Constant {
        match constant {
            rtlola_hir::hir::Constant::Basic(lit) => Self::lower_constant_literal(lit, ty),
            rtlola_hir::hir::Constant::Inlined(Inlined { lit, .. }) => Self::lower_constant_literal(lit, ty),
        }
    }

    fn lower_constant_literal(constant: &rtlola_hir::hir::Literal, ty: &mir::Type) -> mir::Constant {
        match constant {
            rtlola_hir::hir::Literal::Str(s) => mir::Constant::Str(s.clone()),
            rtlola_hir::hir::Literal::Bool(b) => mir::Constant::Bool(*b),
            rtlola_hir::hir::Literal::Integer(i) => {
                match ty {
                    mir::Type::Int(_) => mir::Constant::Int(*i),
                    mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                    _ => unreachable!(),
                }
            },
            rtlola_hir::hir::Literal::SInt(i) => {
                //TODO rewrite to 128 bytes
                match ty {
                    mir::Type::Int(_) => mir::Constant::Int(*i as i64),
                    mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                    _ => unreachable!(),
                }
            },
            rtlola_hir::hir::Literal::Float(f) => mir::Constant::Float(*f),
        }
    }

    fn lower_arith_log_op(op: ArithLogOp) -> mir::ArithLogOp {
        match op {
            ArithLogOp::Not => mir::ArithLogOp::Not,
            ArithLogOp::Neg => mir::ArithLogOp::Neg,
            ArithLogOp::Add => mir::ArithLogOp::Add,
            ArithLogOp::Sub => mir::ArithLogOp::Sub,
            ArithLogOp::Mul => mir::ArithLogOp::Mul,
            ArithLogOp::Div => mir::ArithLogOp::Div,
            ArithLogOp::Rem => mir::ArithLogOp::Rem,
            ArithLogOp::Pow => mir::ArithLogOp::Pow,
            ArithLogOp::And => mir::ArithLogOp::And,
            ArithLogOp::Or => mir::ArithLogOp::Or,
            ArithLogOp::BitXor => mir::ArithLogOp::BitXor,
            ArithLogOp::BitAnd => mir::ArithLogOp::BitAnd,
            ArithLogOp::BitOr => mir::ArithLogOp::BitOr,
            ArithLogOp::BitNot => mir::ArithLogOp::BitNot,
            ArithLogOp::Shl => mir::ArithLogOp::Shr,
            ArithLogOp::Shr => mir::ArithLogOp::Shr,
            ArithLogOp::Eq => mir::ArithLogOp::Eq,
            ArithLogOp::Lt => mir::ArithLogOp::Lt,
            ArithLogOp::Le => mir::ArithLogOp::Le,
            ArithLogOp::Ne => mir::ArithLogOp::Ne,
            ArithLogOp::Ge => mir::ArithLogOp::Ge,
            ArithLogOp::Gt => mir::ArithLogOp::Gt,
        }
    }

    fn lower_window_operation(op: WindowOperation) -> mir::WindowOperation {
        match op {
            WindowOperation::Count => mir::WindowOperation::Count,
            WindowOperation::Min => mir::WindowOperation::Min,
            WindowOperation::Max => mir::WindowOperation::Max,
            WindowOperation::Sum => mir::WindowOperation::Sum,
            WindowOperation::Product => mir::WindowOperation::Product,
            WindowOperation::Average => mir::WindowOperation::Average,
            WindowOperation::Integral => mir::WindowOperation::Integral,
            WindowOperation::Conjunction => mir::WindowOperation::Conjunction,
            WindowOperation::Disjunction => mir::WindowOperation::Disjunction,
        }
    }

    fn lower_stream_access_kind(kind: StreamAccessKind) -> mir::StreamAccessKind {
        match kind {
            StreamAccessKind::Sync => mir::StreamAccessKind::Sync,
            StreamAccessKind::DiscreteWindow(wref) => mir::StreamAccessKind::DiscreteWindow(wref),
            StreamAccessKind::SlidingWindow(wref) => mir::StreamAccessKind::SlidingWindow(wref),
            StreamAccessKind::Hold => mir::StreamAccessKind::Hold,
            StreamAccessKind::Offset(o) => mir::StreamAccessKind::Offset(Self::lower_offset(o)),
        }
    }

    fn lower_offset(offset: Offset) -> mir::Offset {
        match offset {
            Offset::FutureDiscrete(o) => mir::Offset::Future(o),
            Offset::PastDiscrete(o) => mir::Offset::Past(o),
            Offset::FutureRealTime(_) | Offset::PastRealTime(_) => {
                unreachable!("Real-time Lookups should be already transformed to discrete lookups.")
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rtlola_parser::ParserConfig;
    use rtlola_reporting::Handler;

    use super::*;

    fn lower_spec(spec: &str) -> (RtLolaHir<CompleteMode>, mir::RtLolaMir) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let ast = ParserConfig::for_string(spec.into())
            .parse()
            .unwrap_or_else(|e| panic!("{}", e));
        let hir = rtlola_hir::fully_analyzed(ast, &handler).expect("Invalid AST:");
        (hir.clone(), Mir::from_hir(hir))
    }

    #[test]
    fn check_event_based_streams() {
        let spec = "input a: Float64\ninput b:Float64\noutput c := a + b\noutput d := a.hold().defaults(to:0.0) + b\noutput e := a + 9.0\ntrigger d < e\ntrigger a < 5.0";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 2);
        assert_eq!(mir.outputs.len(), 5);
        assert_eq!(mir.event_driven.len(), 5);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 2);
        let hir_a = hir.inputs().find(|i| i.name == "a".to_string()).unwrap();
        let mir_a = mir.inputs.iter().find(|i| i.name == "a".to_string()).unwrap();
        assert_eq!(hir_a.sr(), mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();
        assert_eq!(hir_d.sr(), mir_d.reference);
    }

    #[test]
    fn check_time_driven_streams() {
        let spec = "input a: Int64\ninput b:Int64\noutput c @1Hz:= a.aggregate(over: 2s, using: sum)+ b.aggregate(over: 4s, using:sum)\noutput d @4Hz:= a.hold().defaults(to:0) + b.hold().defaults(to: 0)\noutput e @0.5Hz := a.aggregate(over: 4s, using: sum) + 9\ntrigger d < e";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 2);
        assert_eq!(mir.outputs.len(), 4);
        assert_eq!(mir.event_driven.len(), 0);
        assert_eq!(mir.time_driven.len(), 4);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 3);
        assert_eq!(mir.triggers.len(), 1);
        let hir_a = hir.inputs().find(|i| i.name == "a".to_string()).unwrap();
        let mir_a = mir.inputs.iter().find(|i| i.name == "a".to_string()).unwrap();
        assert_eq!(hir_a.sr(), mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();
        assert_eq!(hir_d.sr(), mir_d.reference);
    }

    #[test]
    #[ignore = "type checker bug"]
    fn check_stream_with_parameter() {
        let spec = "input a: Int8\noutput b(para) spawn with a if a > 6 := a + para\noutput c(para) spawn with a if a > 6 := a + b(para)\noutput d(para) spawn with a if a > 6 := a + c(para)";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 3);
        assert_eq!(mir.event_driven.len(), 3);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 0);
        let hir_a = hir.inputs().find(|i| i.name == "a".to_string()).unwrap();
        let mir_a = mir.inputs.iter().find(|i| i.name == "a".to_string()).unwrap();
        assert_eq!(hir_a.sr(), mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();
        assert_eq!(hir_d.sr(), mir_d.reference);
        todo!()
    }
}
