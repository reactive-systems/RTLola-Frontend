use std::collections::{BTreeSet, HashMap};
use std::iter::zip;

use itertools::Itertools;
use rtlola_hir::hir::{
    ActivationCondition, Aggregation, ArithLogOp, ConcretePacingType, ConcreteValueType, Constant, DepAnaTrait,
    DiscreteAggr, Expression, ExpressionKind, FnExprKind, Inlined, InstanceAggregation, Literal, MemBoundTrait, Offset,
    OrderedTrait, Origin, OutputKind, SlidingAggr, StreamAccessKind, StreamReference, TypedTrait, WidenExprKind,
    Window,
};
use rtlola_hir::{CompleteMode, RtLolaHir};
use rtlola_parser::ast::{InstanceOperation, InstanceSelection, WindowOperation};

use crate::mir::{self, Close, Eval, EvalClause, Mir, PacingLocality, Spawn, Trigger};

impl Mir {
    /// Generates an Mir from a complete Hir.
    pub fn from_hir(hir: RtLolaHir<CompleteMode>) -> Mir {
        let sr_map: HashMap<StreamReference, StreamReference> = hir
            .inputs()
            .sorted_by(|a, b| Ord::cmp(&a.sr(), &b.sr()))
            .enumerate()
            .map(|(new_ref, i)| (i.sr(), StreamReference::In(new_ref)))
            .chain(
                hir.outputs()
                    .sorted_by(|a, b| Ord::cmp(&a.sr(), &b.sr()))
                    .enumerate()
                    .map(|(new_ref, o)| (o.sr(), StreamReference::Out(new_ref))),
            )
            .collect();

        let inputs = hir
            .inputs()
            .sorted_by(|a, b| Ord::cmp(&a.sr(), &b.sr()))
            .map(|i| {
                let sr = i.sr();
                mir::InputStream {
                    name: i.name.clone(),
                    ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                    accessed_by: Self::lower_accessed_streams(&sr_map, hir.direct_accessed_by_with(sr)),
                    aggregated_by: hir
                        .aggregated_by(sr)
                        .into_iter()
                        .map(|(sr, wr)| (sr_map[&sr], wr))
                        .collect(),
                    layer: hir.stream_layers(sr),
                    memory_bound: hir.memory_bound(sr),
                    reference: sr_map[&sr],
                }
            })
            .collect::<Vec<mir::InputStream>>();
        assert!(
            inputs.iter().enumerate().all(|(idx, i)| idx == i.reference.in_ix()),
            "SRefs need to enumerated from 0 to the number of streams"
        );

        let outputs = hir.outputs().map(|o| {
            let sr = o.sr();
            mir::OutputStream {
                name: o.name(),
                kind: o.kind.clone(),
                ty: Self::lower_value_type(&hir.stream_type(sr).value_ty),
                spawn: Self::lower_spawn(&hir, &sr_map, sr),
                eval: Self::lower_eval(&hir, &sr_map, sr),
                close: Self::lower_close(&hir, &sr_map, sr),
                accesses: Self::lower_accessed_streams(&sr_map, hir.direct_accesses_with(sr)),
                accessed_by: Self::lower_accessed_streams(&sr_map, hir.direct_accessed_by_with(sr)),
                aggregated_by: hir
                    .aggregated_by(sr)
                    .into_iter()
                    .map(|(sr, wr)| (sr_map[&sr], wr))
                    .collect(),
                memory_bound: hir.memory_bound(sr),
                layer: hir.stream_layers(sr),
                reference: sr_map[&sr],
                params: Self::lower_parameters(&hir, sr),
            }
        });

        let outputs = outputs
            .sorted_by(|a, b| Ord::cmp(&a.reference, &b.reference))
            .collect::<Vec<_>>();

        assert!(
            outputs.iter().enumerate().all(|(idx, o)| idx == o.reference.out_ix()),
            "SRefs need to enumerated from 0 to the number of streams"
        );

        let time_driven = outputs
            .iter()
            .filter(|o| hir.is_periodic(o.reference))
            .map(|o| Self::lower_periodic(&hir, &sr_map, o.reference))
            .collect::<Vec<mir::TimeDrivenStream>>();
        let event_driven = outputs
            .iter()
            .filter(|o| hir.is_event(o.reference))
            .map(|o| Self::lower_event_based(&hir, &sr_map, o.reference))
            .collect::<Vec<mir::EventDrivenStream>>();

        let discrete_windows = hir
            .discrete_windows()
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.reference().idx(), &b.reference().idx()))
            .map(|win| Self::lower_discrete_window(&hir, &sr_map, win))
            .collect::<Vec<mir::DiscreteWindow>>();

        assert!(
            discrete_windows
                .iter()
                .enumerate()
                .all(|(idx, w)| idx == w.reference.idx()),
            "WRefs need to enumerate from 0 to the number of discrete windows"
        );

        let sliding_windows = hir
            .sliding_windows()
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.reference().idx(), &b.reference().idx()))
            .map(|win| Self::lower_sliding_window(&hir, &sr_map, win))
            .collect::<Vec<mir::SlidingWindow>>();
        assert!(
            sliding_windows
                .iter()
                .enumerate()
                .all(|(idx, w)| idx == w.reference.idx()),
            "WRefs need to enumerate from 0 to the number of sliding windows"
        );

        let instance_aggregations = hir
            .instance_aggregations()
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.reference().idx(), &b.reference().idx()))
            .map(|win| Self::lower_instance_aggregation(&hir, &sr_map, win))
            .collect::<Vec<mir::InstanceAggregation>>();
        assert!(
            instance_aggregations
                .iter()
                .enumerate()
                .all(|(idx, w)| idx == w.reference.idx()),
            "WRefs need to enumerate from 0 to the number of discrete windows"
        );

        let triggers = outputs
            .iter()
            .filter_map(|output| matches!(&output.kind, OutputKind::Trigger(_)).then_some(output.reference))
            .enumerate()
            .map(|(trigger_reference, output_reference)| {
                Trigger {
                    trigger_reference,
                    output_reference,
                }
            })
            .collect();

        Mir {
            inputs,
            outputs,
            time_driven,
            event_driven,
            discrete_windows,
            sliding_windows,
            instance_aggregations,
            triggers,
        }
    }

    fn lower_event_based(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        sr: StreamReference,
    ) -> mir::EventDrivenStream {
        if let ConcretePacingType::Event(ac) = hir.stream_type(sr).eval_pacing {
            mir::EventDrivenStream {
                reference: sr_map[&sr],
                ac: Self::lower_activation_condition(&ac, sr_map),
            }
        } else {
            unreachable!()
        }
    }

    fn lower_activation_condition(
        ac: &ActivationCondition,
        sr_map: &HashMap<StreamReference, StreamReference>,
    ) -> mir::ActivationCondition {
        let lower_conjunction = |conjs: &BTreeSet<StreamReference>| -> mir::ActivationCondition {
            if conjs.len() == 1 {
                let sref = conjs.iter().next().unwrap();
                mir::ActivationCondition::Stream(sr_map[sref])
            } else {
                mir::ActivationCondition::Conjunction(
                    conjs
                        .iter()
                        .map(|sr| mir::ActivationCondition::Stream(sr_map[sr]))
                        .collect(),
                )
            }
        };

        match ac {
            ActivationCondition::Models(disjuncts) if disjuncts.len() == 1 => {
                let conj = disjuncts.iter().next().unwrap();
                lower_conjunction(conj)
            },
            ActivationCondition::Models(disjuncts) => {
                mir::ActivationCondition::Disjunction(disjuncts.iter().map(lower_conjunction).collect())
            },
            ActivationCondition::True => mir::ActivationCondition::True,
        }
    }

    fn lower_periodic(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        sr: StreamReference,
    ) -> mir::TimeDrivenStream {
        match &hir.stream_type(sr).eval_pacing {
            ConcretePacingType::FixedGlobalPeriodic(f) => {
                mir::TimeDrivenStream {
                    reference: sr_map[&sr],
                    frequency: *f,
                    locality: PacingLocality::Global,
                }
            },
            ConcretePacingType::FixedLocalPeriodic(f) => {
                mir::TimeDrivenStream {
                    reference: sr_map[&sr],
                    frequency: *f,
                    locality: PacingLocality::Local,
                }
            },
            _ => unreachable!(),
        }
    }

    fn lower_pacing_type(
        cpt: ConcretePacingType,
        sr_map: &HashMap<StreamReference, StreamReference>,
    ) -> mir::PacingType {
        match cpt {
            ConcretePacingType::Event(ac) => mir::PacingType::Event(Self::lower_activation_condition(&ac, sr_map)),
            ConcretePacingType::FixedLocalPeriodic(freq) => mir::PacingType::LocalPeriodic(freq),
            ConcretePacingType::FixedGlobalPeriodic(freq) => mir::PacingType::GlobalPeriodic(freq),
            ConcretePacingType::Constant => mir::PacingType::Constant,
            other => {
                unreachable!("Ensured by pacing type checker: {:?}", other)
            },
        }
    }

    fn lower_spawn(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        sr: StreamReference,
    ) -> Spawn {
        let ty = hir.stream_type(sr);
        let spawn_pacing = Self::lower_pacing_type(ty.spawn_pacing, sr_map);
        let hir_spawn_expr = hir.spawn_expr(sr);
        let hir_spawn_condition = hir.spawn_cond(sr);
        let spawn_cond = hir_spawn_condition.map(|expr| Self::lower_expr(hir, sr_map, expr));
        let spawn_expression = hir_spawn_expr.map(|expr| Self::lower_expr(hir, sr_map, expr));
        Spawn {
            expression: spawn_expression,
            pacing: spawn_pacing,
            condition: spawn_cond,
        }
    }

    fn lower_eval(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        sr: StreamReference,
    ) -> Eval {
        assert_eq!(hir.eval_expr(sr).unwrap().len(), hir.eval_cond(sr).unwrap().len());

        let clauses = zip(hir.eval_expr(sr).unwrap(), hir.eval_cond(sr).unwrap())
            .enumerate()
            .map(|(idx, (expr, cond))| {
                let expr = Self::lower_expr(hir, sr_map, expr);
                let condition = cond.map(|f| Self::lower_expr(hir, sr_map, f));
                let pacing = Self::lower_pacing_type(hir.eval_pacing_type(sr, idx), sr_map);
                EvalClause {
                    pacing,
                    condition,
                    expression: expr,
                }
            })
            .collect();

        let eval_pacing = Self::lower_pacing_type(hir.stream_type(sr).eval_pacing, sr_map);
        Eval { clauses, eval_pacing }
    }

    fn lower_close(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        sr: StreamReference,
    ) -> Close {
        let (close, close_pacing, close_self_ref) = hir
            .close_cond(sr)
            .map(|expr| {
                let cpt = hir.stream_type(sr).close_pacing;
                let close_self_ref = matches!(
                    hir.expr_type(expr.id()).spawn_pacing,
                    ConcretePacingType::Event(_)
                        | ConcretePacingType::FixedGlobalPeriodic(_)
                        | ConcretePacingType::FixedLocalPeriodic(_)
                );
                (
                    Some(Self::lower_expr(hir, sr_map, expr)),
                    Self::lower_pacing_type(cpt, sr_map),
                    close_self_ref,
                )
            })
            .unwrap_or((None, mir::PacingType::Constant, false));
        Close {
            condition: close,
            pacing: close_pacing,
            has_self_reference: close_self_ref,
        }
    }

    fn lower_sliding_window(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        win: &Window<SlidingAggr>,
    ) -> mir::SlidingWindow {
        mir::SlidingWindow {
            target: sr_map[&win.target],
            caller: sr_map[&win.caller],
            duration: win.aggr.duration,
            num_buckets: hir.num_buckets(win.reference()),
            bucket_size: hir.bucket_size(win.reference()),
            wait: win.aggr.wait,
            op: Self::lower_window_operation(win.aggr.op),
            reference: win.reference(),
            ty: Self::lower_value_type(&hir.expr_type(win.id()).value_ty),
        }
    }

    fn lower_discrete_window(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        win: &Window<DiscreteAggr>,
    ) -> mir::DiscreteWindow {
        mir::DiscreteWindow {
            target: sr_map[&win.target],
            caller: sr_map[&win.caller],
            duration: win.aggr.duration,
            wait: win.aggr.wait,
            op: Self::lower_window_operation(win.aggr.op),
            reference: win.reference(),
            ty: Self::lower_value_type(&hir.expr_type(win.id()).value_ty),
        }
    }

    fn lower_instance_aggregation(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        win: &InstanceAggregation,
    ) -> mir::InstanceAggregation {
        mir::InstanceAggregation {
            target: sr_map[&win.target],
            caller: sr_map[&win.caller],
            reference: win.reference(),
            selection: Self::lower_instance_selection(win.selection),
            aggr: Self::lower_instance_operation(win.aggr),
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

    fn lower_expr(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        expr: &Expression,
    ) -> mir::Expression {
        let ty = Self::lower_value_type(&hir.expr_type(expr.id()).value_ty);
        mir::Expression {
            kind: Self::lower_expression_kind(hir, sr_map, &expr.kind, &ty),
            ty,
        }
    }

    fn lower_expression_kind(
        hir: &RtLolaHir<CompleteMode>,
        sr_map: &HashMap<StreamReference, StreamReference>,
        expr: &ExpressionKind,
        ty: &mir::Type,
    ) -> mir::ExpressionKind {
        match expr {
            ExpressionKind::LoadConstant(constant) => {
                mir::ExpressionKind::LoadConstant(Self::lower_constant(constant, ty))
            },
            ExpressionKind::ArithLog(op, args) => {
                let op = Self::lower_arith_log_op(*op);
                let args = args
                    .iter()
                    .map(|arg| Self::lower_expr(hir, sr_map, arg))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::ArithLog(op, args)
            },
            ExpressionKind::StreamAccess(sr, kind, para) => {
                mir::ExpressionKind::StreamAccess {
                    target: sr_map[sr],
                    access_kind: Self::lower_stream_access_kind(*kind),
                    parameters: para.iter().map(|p| Self::lower_expr(hir, sr_map, p)).collect(),
                }
            },
            ExpressionKind::ParameterAccess(sr, para) => mir::ExpressionKind::ParameterAccess(sr_map[sr], *para),
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let condition = Box::new(Self::lower_expr(hir, sr_map, condition));
                let consequence = Box::new(Self::lower_expr(hir, sr_map, consequence));
                let alternative = Box::new(Self::lower_expr(hir, sr_map, alternative));
                mir::ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                }
            },
            ExpressionKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| Self::lower_expr(hir, sr_map, element))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Tuple(elements)
            },
            ExpressionKind::TupleAccess(tuple, element_pos) => {
                let tuple = Box::new(Self::lower_expr(hir, sr_map, tuple));
                let element_pos = *element_pos;
                mir::ExpressionKind::TupleAccess(tuple, element_pos)
            },
            ExpressionKind::Function(kind) => {
                let FnExprKind { name, args, .. } = kind;
                match name.as_ref() {
                    "cast" => {
                        assert_eq!(args.len(), 1);
                        let expr = Box::new(Self::lower_expr(hir, sr_map, &args[0]));
                        mir::ExpressionKind::Convert { expr }
                    },
                    _ => {
                        let args = args
                            .iter()
                            .map(|arg| Self::lower_expr(hir, sr_map, arg))
                            .collect::<Vec<mir::Expression>>();
                        mir::ExpressionKind::Function(name.clone(), args)
                    },
                }
            },
            ExpressionKind::Widen(kind) => {
                let WidenExprKind { expr, .. } = kind;
                let expr = Box::new(Self::lower_expr(hir, sr_map, expr));
                mir::ExpressionKind::Convert { expr }
            },
            ExpressionKind::Default { expr, default } => {
                let expr = Box::new(Self::lower_expr(hir, sr_map, expr));
                let default = Box::new(Self::lower_expr(hir, sr_map, default));
                mir::ExpressionKind::Default { expr, default }
            },
        }
    }

    fn lower_constant(constant: &Constant, ty: &mir::Type) -> mir::Constant {
        match constant {
            Constant::Basic(lit) => Self::lower_constant_literal(lit, ty),
            Constant::Inlined(Inlined { lit, .. }) => Self::lower_constant_literal(lit, ty),
        }
    }

    fn lower_constant_literal(constant: &Literal, ty: &mir::Type) -> mir::Constant {
        match constant {
            Literal::Str(s) => mir::Constant::Str(s.clone()),
            Literal::Bool(b) => mir::Constant::Bool(*b),
            Literal::Integer(i) => {
                match ty {
                    mir::Type::Int(_) => mir::Constant::Int(*i),
                    mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                    _ => unreachable!(),
                }
            },
            Literal::SInt(i) => {
                //TODO rewrite to 128 bytes
                match ty {
                    mir::Type::Int(_) => mir::Constant::Int(*i as i64),
                    mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                    _ => unreachable!(),
                }
            },
            Literal::Float(f) => mir::Constant::Float(*f),
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
            ArithLogOp::Shl => mir::ArithLogOp::Shl,
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

    fn lower_instance_operation(op: InstanceOperation) -> mir::InstanceOperation {
        match op {
            InstanceOperation::Count => mir::InstanceOperation::Count,
            InstanceOperation::Min => mir::InstanceOperation::Min,
            InstanceOperation::Max => mir::InstanceOperation::Max,
            InstanceOperation::Sum => mir::InstanceOperation::Sum,
            InstanceOperation::Product => mir::InstanceOperation::Product,
            InstanceOperation::Average => mir::InstanceOperation::Average,
            InstanceOperation::Conjunction => mir::InstanceOperation::Conjunction,
            InstanceOperation::Disjunction => mir::InstanceOperation::Disjunction,
            InstanceOperation::Variance => mir::InstanceOperation::Variance,
            InstanceOperation::Covariance => mir::InstanceOperation::Covariance,
            InstanceOperation::StandardDeviation => mir::InstanceOperation::StandardDeviation,
            InstanceOperation::NthPercentile(x) => mir::InstanceOperation::NthPercentile(x),
        }
    }

    fn lower_instance_selection(sel: InstanceSelection) -> mir::InstanceSelection {
        match sel {
            InstanceSelection::Fresh => mir::InstanceSelection::Fresh,
            InstanceSelection::All => mir::InstanceSelection::All,
        }
    }

    fn lower_stream_access_kind(kind: StreamAccessKind) -> mir::StreamAccessKind {
        match kind {
            StreamAccessKind::Sync => mir::StreamAccessKind::Sync,
            StreamAccessKind::DiscreteWindow(wref) => mir::StreamAccessKind::DiscreteWindow(wref),
            StreamAccessKind::SlidingWindow(wref) => mir::StreamAccessKind::SlidingWindow(wref),
            StreamAccessKind::InstanceAggregation(wref) => mir::StreamAccessKind::InstanceAggregation(wref),
            StreamAccessKind::Hold => mir::StreamAccessKind::Hold,
            StreamAccessKind::Offset(o) => mir::StreamAccessKind::Offset(Self::lower_offset(o)),
            StreamAccessKind::Get => mir::StreamAccessKind::Get,
            StreamAccessKind::Fresh => mir::StreamAccessKind::Fresh,
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

    fn lower_accessed_streams(
        sr_map: &HashMap<StreamReference, StreamReference>,
        streams: Vec<(StreamReference, Vec<(Origin, StreamAccessKind)>)>,
    ) -> Vec<(StreamReference, Vec<(Origin, mir::StreamAccessKind)>)> {
        streams
            .into_iter()
            .map(|(sref, kinds)| {
                (
                    sr_map[&sref],
                    kinds
                        .into_iter()
                        .map(|(origin, kind)| (origin, Self::lower_stream_access_kind(kind)))
                        .collect(),
                )
            })
            .collect()
    }

    fn lower_parameters(hir: &RtLolaHir<CompleteMode>, sr: StreamReference) -> Vec<mir::Parameter> {
        let params = hir.output(sr).expect("is output stream").params();
        params
            .map(|parameter| {
                mir::Parameter {
                    name: parameter.name.clone(),
                    ty: Self::lower_value_type(&hir.get_parameter_type(sr, parameter.index())),
                    idx: parameter.index(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use rtlola_parser::ParserConfig;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    use super::*;
    use crate::mir::IntTy::Int8;
    use crate::mir::{PacingType, Stream};

    fn lower_spec(spec: &str) -> (RtLolaHir<CompleteMode>, mir::RtLolaMir) {
        let ast = ParserConfig::for_string(spec.into())
            .parse()
            .unwrap_or_else(|e| panic!("{:?}", e));
        let hir = rtlola_hir::fully_analyzed(ast).expect("Invalid AST:");
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
        let hir_d = hir.outputs().find(|i| i.name() == "d".to_string()).unwrap();
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
        let hir_d = hir.outputs().find(|i| i.name() == "d".to_string()).unwrap();
        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();
        assert_eq!(hir_d.sr(), mir_d.reference);
    }

    #[test]
    fn check_stream_with_parameter() {
        let spec = "input a: Int8\n\
        output d(para) spawn with a when a > 6 eval @a with para";
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
        let hir_d = hir.outputs().find(|i| i.name() == "d".to_string()).unwrap();
        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();
        assert_eq!(hir_d.sr(), mir_d.reference);
        assert_eq!(
            &mir_d.spawn.expression,
            &Some(mir::Expression {
                kind: mir::ExpressionKind::StreamAccess {
                    target: mir_a.reference,
                    parameters: vec![],
                    access_kind: mir::StreamAccessKind::Sync,
                },
                ty: mir::Type::Int(Int8),
            })
        );
        assert!(matches!(
            &mir_d.spawn.condition,
            &Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Gt, _),
                ty: _,
            })
        ));
        assert_eq!(
            &mir_d.spawn.pacing,
            &mir::PacingType::Event(mir::ActivationCondition::Stream(mir_a.reference))
        );
        assert_eq!(
            &mir_d.eval.clauses[0].expression,
            &mir::Expression {
                kind: mir::ExpressionKind::ParameterAccess(mir_d.reference, 0),
                ty: mir::Type::Int(Int8),
            }
        );
    }

    #[test]
    fn check_spawn_filter_close() {
        let spec = "input a: Int8\n\
        output d spawn @1Hz when a.hold().defaults(to:0) > 6 eval @a when a = 42 with a close when a = 1337";
        let (_, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 1);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.event_driven.len(), 1);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 0);

        let mir_d = mir.outputs.iter().find(|i| i.name == "d".to_string()).unwrap();

        assert!(mir_d.spawn.expression.is_none());
        assert!(matches!(
            &mir_d.spawn.condition,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Gt, _),
                ty: _,
            })
        ));
        assert_eq!(
            mir_d.spawn.pacing,
            mir::PacingType::GlobalPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
        assert!(matches!(
            mir_d.eval.clauses[0].condition,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Eq, _),
                ty: _,
            })
        ));
        assert!(matches!(
            mir_d.close.condition,
            Some(mir::Expression {
                kind: mir::ExpressionKind::ArithLog(mir::ArithLogOp::Eq, _),
                ty: _,
            })
        ));
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
    fn test_instance_window() {
        let spec = "input a: Int32\n\
        output b(p: Bool) spawn with a == 42 eval with a\n\
        output c @1Hz := b(false).aggregate(over: 1s, using: sum)";
        let (_, mir) = lower_spec(spec);

        let expr = mir.outputs[1].eval.clauses[0].expression.kind.clone();
        assert!(
            matches!(expr, mir::ExpressionKind::StreamAccess {target: _, parameters: paras, access_kind: mir::StreamAccessKind::SlidingWindow(_)} if paras.len() == 1)
        );
    }

    #[test]
    fn test_cast_lowering() {
        let spec = "input a: Int64\n\
        output b := cast<Int64, Float64>(a)";
        let (_, mir) = lower_spec(spec);
        assert!(matches!(
            mir.outputs[0].eval.clauses[0].expression.kind,
            mir::ExpressionKind::Convert { expr: _ }
        ));
    }

    #[test]
    fn test_multiple_eval_clauses() {
        let spec = "input a: Int64\ninput b: Int64\ninput c: Bool\n\
        output d eval @(c&&a) when c with a eval @(c&&b) when !c with b";
        let (_, mir) = lower_spec(spec);
        assert_eq!(mir.outputs.len(), 1);
        assert_eq!(mir.inputs.len(), 3);
        assert_eq!(mir.triggers.len(), 0);

        let output = &mir.outputs[0];
        assert_eq!(output.eval.clauses.len(), 2);
        assert_eq!(
            output.eval.eval_pacing,
            PacingType::Event(mir::ActivationCondition::Disjunction(vec![
                mir::ActivationCondition::Conjunction(vec![
                    // TODO:??
                    mir::ActivationCondition::Stream(StreamReference::In(0)),
                    mir::ActivationCondition::Stream(StreamReference::In(1)),
                    mir::ActivationCondition::Stream(StreamReference::In(2)),
                ]),
                mir::ActivationCondition::Conjunction(vec![
                    mir::ActivationCondition::Stream(StreamReference::In(0)),
                    mir::ActivationCondition::Stream(StreamReference::In(2)),
                ]),
                mir::ActivationCondition::Conjunction(vec![
                    mir::ActivationCondition::Stream(StreamReference::In(1)),
                    mir::ActivationCondition::Stream(StreamReference::In(2)),
                ]),
            ]))
        );
        assert_eq!(
            output.eval.clauses[0].pacing,
            PacingType::Event(mir::ActivationCondition::Conjunction(vec![
                mir::ActivationCondition::Stream(StreamReference::In(0)),
                mir::ActivationCondition::Stream(StreamReference::In(2)),
            ]))
        );
        assert_eq!(
            output.eval.clauses[1].pacing,
            PacingType::Event(mir::ActivationCondition::Conjunction(vec![
                mir::ActivationCondition::Stream(StreamReference::In(1)),
                mir::ActivationCondition::Stream(StreamReference::In(2)),
            ]))
        );
    }

    #[test]
    fn spawn_close_ac_only() {
        let spec = "input a : UInt64\noutput b spawn @a eval @true with true";
        let (_, mir) = lower_spec(spec);
        assert!(mir.outputs[0].is_spawned());

        let spec = "input a : UInt64\noutput b spawn when a == 0 eval @Global(1Hz) with true close @Local(5s)";
        let (_, mir) = lower_spec(spec);
        assert!(mir.outputs[0].is_closed());
    }
}
