use itertools::Itertools;
use rtlola_hir::hir::{
    ActivationCondition, ArithLogOp, ConcretePacingType, ConcreteValueType, Constant, DepAnaTrait, DiscreteAggr,
    Expression, ExpressionKind, FnExprKind, Inlined, MemBoundTrait, Offset, OrderedTrait, SlidingAggr,
    StreamAccessKind, StreamReference, TypedTrait, WidenExprKind, Window,
};
use rtlola_hir::{CompleteMode, RtLolaHir};
use rtlola_parser::ast::WindowOperation;

use crate::mir;
use crate::mir::{InstanceTemplate, Mir, PacingType, SpawnTemplate};

impl Mir {
    /// Generates an Mir from a complete Hir.
    pub(crate) fn from_hir(hir: RtLolaHir<CompleteMode>) -> Mir {
        let inputs = hir
            .inputs()
            .sorted_by(|a, b| Ord::cmp(&a.sr(), &b.sr()))
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
        assert!(
            inputs.iter().enumerate().all(|(idx, i)| idx == i.reference.in_ix()),
            "SRefs need to enumerated from 0 to the number of streams"
        );

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
                    info_streams: t.info_streams.clone(),
                    reference: sr,
                    trigger_reference: index,
                };
                let mir_output_stream = mir::OutputStream {
                    name: format!("trigger_{}", index), //TODO better name
                    ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                    expr: Self::lower_expr(&hir, hir.expr(sr)),
                    instance_template: InstanceTemplate::default(),
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

        assert!(
            outputs.iter().enumerate().all(|(idx, o)| idx == o.reference.out_ix()),
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
            .sorted_by(|a, b| Ord::cmp(&a.reference().idx(), &b.reference().idx()))
            .map(|win| Self::lower_discrete_window(&hir, win))
            .collect::<Vec<mir::DiscreteWindow>>();
        assert!(
            discrete_windows
                .iter()
                .enumerate()
                .all(|(idx, w)| idx == w.reference.idx()),
            "WRefs need to enumerated from 0 to the number of discrete windows"
        );

        let sliding_windows = hir
            .sliding_windows()
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.reference().idx(), &b.reference().idx()))
            .map(|win| Self::lower_sliding_window(&hir, win))
            .collect::<Vec<mir::SlidingWindow>>();
        assert!(
            sliding_windows
                .iter()
                .enumerate()
                .all(|(idx, w)| idx == w.reference.idx()),
            "WRefs need to enumerated from 0 to the number of sliding windows"
        );

        Mir {
            inputs,
            outputs,
            time_driven,
            event_driven,
            discrete_windows,
            sliding_windows,
            triggers,
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

    fn lower_instance_template(hir: &RtLolaHir<CompleteMode>, sr: StreamReference) -> InstanceTemplate {
        let rtlola_hir::hir::StreamType {
            value_ty: _,
            pacing_ty: _,
            spawn,
            filter,
            close,
        } = hir.stream_type(sr);
        let spawn_pacing = match &spawn.0 {
            ConcretePacingType::Event(ac) => PacingType::Event(Self::lower_activation_condition(ac)),
            ConcretePacingType::FixedPeriodic(freq) => PacingType::Periodic(*freq),
            ConcretePacingType::Constant => PacingType::Constant,
            ConcretePacingType::Periodic => {
                unreachable!("Ensured by pacing type checker")
            },
        };
        let (hir_spawn_target, hir_spawn_condition) = hir.spawn(sr).unwrap_or((None, None));
        let spawn_cond = hir_spawn_condition.map(|_| Self::lower_expr(hir, &spawn.1));
        let spawn_target = hir_spawn_target.map(|target| Self::lower_expr(hir, target));
        let filter = hir.filter(sr).map(|_| Self::lower_expr(hir, &filter));
        let close = hir.close(sr).map(|_| Self::lower_expr(hir, &close));
        InstanceTemplate {
            spawn: SpawnTemplate {
                target: spawn_target,
                pacing: spawn_pacing,
                condition: spawn_cond,
            },
            filter,
            close,
        }
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
            rtlola_hir::hir::ExpressionKind::ParameterAccess(sr, para) => {
                mir::ExpressionKind::ParameterAccess(*sr, *para)
            },
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
                match name.as_ref() {
                    "cast" => {
                        assert_eq!(args.len(), 1);
                        let expr = Box::new(Self::lower_expr(hir, &args[0]));
                        mir::ExpressionKind::Convert { expr }
                    },
                    _ => {
                        let args = args
                            .iter()
                            .map(|arg| Self::lower_expr(hir, arg))
                            .collect::<Vec<mir::Expression>>();
                        mir::ExpressionKind::Function(name.clone(), args)
                    },
                }
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
            WindowOperation::Last => mir::WindowOperation::Last,
            WindowOperation::Variance => mir::WindowOperation::Variance,
            WindowOperation::Covariance => mir::WindowOperation::Covariance,
            WindowOperation::StandardDeviation => mir::WindowOperation::StandardDeviation,
            WindowOperation::NthPercentile(x) => mir::WindowOperation::NthPercentile(x),
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

    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use rtlola_parser::ParserConfig;
    use rtlola_reporting::Handler;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    use super::*;
    use crate::mir::IntTy::Int8;

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
    fn check_stream_with_parameter() {
        let spec = "input a: Int8\n\
        output d(para) @a spawn with a if a > 6 := para";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.event_driven.len(), 1);
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
        assert_eq!(
            &mir_d.instance_template.spawn.target,
            &Some(mir::Expression {
                kind: mir::ExpressionKind::StreamAccess {
                    target: mir_a.reference,
                    parameters: vec![],
                    access_kind: mir::StreamAccessKind::Sync,
                },
                ty: mir::Type::Int(Int8)
            })
        );
        assert!(matches!(
            &mir_d.instance_template.spawn.condition,
            &Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Gt, _),
                ty: _,
            })
        ));
        assert_eq!(
            &mir_d.instance_template.spawn.pacing,
            &PacingType::Event(mir::ActivationCondition::Stream(mir_a.reference))
        );
        assert_eq!(
            &mir_d.expr,
            &mir::Expression {
                kind: mir::ExpressionKind::ParameterAccess(mir_d.reference, 0),
                ty: mir::Type::Int(Int8)
            }
        );
    }

    #[test]
    fn check_spawn_filter_close() {
        let spec = "input a: Int8\n\
        output d @a spawn @1Hz if a.hold().defaults(to:0) > 6 filter a = 42 close a = 1337 := a";
        let (_, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.event_driven.len(), 1);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 0);

        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();

        assert!(mir_d.instance_template.spawn.target.is_none());
        assert!(matches!(
            &mir_d.instance_template.spawn.condition,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Gt, _),
                ty: _,
            })
        ));
        assert_eq!(
            mir_d.instance_template.spawn.pacing,
            PacingType::Periodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
        assert!(matches!(
            mir_d.instance_template.filter,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Eq, _),
                ty: _,
            })
        ));
        assert!(matches!(
            mir_d.instance_template.close,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Eq, _),
                ty: _,
            })
        ));
    }

    #[test]
    fn test_trigger_with_info() {
        let spec = "input a: Bool\n\
        trigger a \"test message\" (a)";
        let (_, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.event_driven.len(), 1);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 1);
        let trigger = mir.triggers[0].clone();
        assert_eq!(trigger.info_streams[0], StreamReference::In(0));

        assert_eq!(trigger.message, "test message");
    }

    #[test]
    fn test_periodic_trigger() {
        let spec = "input a: Bool\n\
        trigger @1Hz a.hold(or: false)";
        let (_, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.event_driven.len(), 0);
        assert_eq!(mir.time_driven.len(), 1);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 1);
    }

    #[test]
    fn test_cast_lowering() {
        let spec = "input a: Int64\n\
        output b := cast<Int64, Float64>(a)";
        let (_, mir) = lower_spec(spec);
        assert!(matches!(
            mir.outputs[0].expr.kind,
            mir::ExpressionKind::Convert { expr: _ }
        ));
    }
}
