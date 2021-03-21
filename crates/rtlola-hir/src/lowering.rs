impl Mir {
    pub(crate) fn from_hir(self) -> Mir {
        let Hir {
            inputs,
            outputs,
            triggers,
            mode,
            ..
        } = &self;
        let inputs = inputs
            .iter()
            .map(|i| {
                let sr = i.sr;
                mir::InputStream {
                    name: i.name.clone(),
                    ty: Self::lower_value_type(self.stream_type(sr).get_value_type()),
                    acccessed_by: mode.direct_accesses(sr),
                    aggregated_by: mode.aggregated_by(sr),
                    layer: mode.stream_layers(sr),
                    memory_bound: mode.memory_bound(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::InputStream>>();
        // assert that each sr is available
        let outputs = outputs
            .iter()
            .map(|o| {
                let sr = o.sr;
                mir::OutputStream {
                    name: o.name.clone(),
                    ty: Self::lower_value_type(self.stream_type(sr).get_value_type()),
                    expr: self.lower_expr(self.expr(sr)),
                    instance_template: self.lower_instance_template(sr),
                    acccesses: mode.direct_accesses(sr),
                    acccessed_by: mode.direct_accessed_by(sr),
                    aggregated_by: mode.aggregated_by(sr),
                    memory_bound: mode.memory_bound(sr),
                    layer: mode.stream_layers(sr),
                    reference: sr,
                }
            })
            .collect::<Vec<mir::OutputStream>>();
        let (trigger_streams, triggers): (Vec<mir::OutputStream>, Vec<mir::Trigger>) = triggers
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.sr, &b.sr))
            .enumerate()
            .map(|(index, t)| {
                let sr = t.sr;
                let mir_trigger = mir::Trigger {
                    message: t.message.clone(),
                    reference: sr,
                    trigger_reference: index,
                };
                let mir_output_stream = mir::OutputStream {
                    name: format!("trigger_{}", index), //TODO better name
                    ty: Self::lower_value_type(self.stream_type(sr).get_value_type()),
                    expr: self.lower_expr(self.expr(sr)),
                    instance_template: None,
                    acccesses: mode.direct_accesses(sr),
                    acccessed_by: mode.direct_accessed_by(sr),
                    aggregated_by: mode.aggregated_by(sr),
                    memory_bound: mode.memory_bound(sr),
                    layer: mode.stream_layers(sr),
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
            "Streamreferences need to enumerated from 0 to the number of streams"
        );
        let time_driven = outputs
            .iter()
            .filter(|o| mode.is_periodic(o.reference))
            .map(|o| self.lower_periodic(o.reference))
            .collect::<Vec<mir::TimeDrivenStream>>();
        let event_driven = outputs
            .iter()
            .filter(|o| mode.is_event(o.reference))
            .map(|o| self.lower_event_based(o.reference))
            .collect::<Vec<mir::EventDrivenStream>>();

        let (sliding_windows, discrete_windows) = mode.all_windows();
        let discrete_windows = discrete_windows
            .into_iter()
            .map(|win| self.lower_discrete_window(win))
            .collect::<Vec<mir::DiscreteWindow>>();
        let sliding_windows = sliding_windows
            .into_iter()
            .map(|win| self.lower_sliding_window(win))
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

    fn lower_event_based(&self, sr: StreamReference) -> mir::EventDrivenStream {
        if let PacingTy::Event(ac) = self.stream_type(sr).get_pacing_type() {
            mir::EventDrivenStream {
                reference: sr,
                ac: Self::lower_activation_condition(ac),
            }
        } else {
            unreachable!()
        }
    }

    fn lower_activation_condition(ac: &ActivationCondition) -> mir::ActivationCondition {
        match ac {
            ActivationCondition::Conjunction(con) => mir::ActivationCondition::Conjunction(
                con.iter().map(Self::lower_activation_condition).collect(),
            ),
            ActivationCondition::Disjunction(dis) => mir::ActivationCondition::Disjunction(
                dis.iter().map(Self::lower_activation_condition).collect(),
            ),
            ActivationCondition::Stream(sr) => mir::ActivationCondition::Stream(*sr),
            ActivationCondition::True => mir::ActivationCondition::True,
        }
    }

    fn lower_periodic(&self, sr: StreamReference) -> mir::TimeDrivenStream {
        if let PacingTy::FixedPeriodic(freq) = self.stream_type(sr).get_pacing_type() {
            mir::TimeDrivenStream {
                reference: sr,
                frequency: *freq,
            }
        } else {
            unreachable!()
        }
    }

    fn lower_instance_template(&self, _sr: StreamReference) -> Option<InstanceTemplate> {
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

    fn lower_sliding_window(&self, win: hir::SlidingWindow) -> mir::SlidingWindow {
        mir::SlidingWindow {
            target: win.target,
            caller: win.caller,
            duration: win.duration,
            wait: win.wait,
            op: win.op,
            reference: win.reference,
            ty: Self::lower_value_type(self.expr_type(win.eid).get_value_type()),
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
            ty: Self::lower_value_type(self.expr_type(win.eid).get_value_type()),
        }
    }

    fn lower_value_type(ty: &ValueTy) -> mir::Type {
        match ty {
            ValueTy::Bool => mir::Type::Bool,
            ValueTy::Integer8 => mir::Type::Int(mir::IntTy::I8),
            ValueTy::Integer16 => mir::Type::Int(mir::IntTy::I16),
            ValueTy::Integer32 => mir::Type::Int(mir::IntTy::I32),
            ValueTy::Integer64 => mir::Type::Int(mir::IntTy::I64),
            ValueTy::UInteger8 => mir::Type::UInt(mir::UIntTy::U8),
            ValueTy::UInteger16 => mir::Type::UInt(mir::UIntTy::U16),
            ValueTy::UInteger32 => mir::Type::UInt(mir::UIntTy::U32),
            ValueTy::UInteger64 => mir::Type::UInt(mir::UIntTy::U64),
            ValueTy::Float32 => mir::Type::Float(mir::FloatTy::F32),
            ValueTy::Float64 => mir::Type::Float(mir::FloatTy::F64),
            ValueTy::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(Self::lower_value_type)
                    .collect::<Vec<_>>();
                mir::Type::Tuple(elements)
            }
            ValueTy::TString => mir::Type::String,
            ValueTy::Byte => mir::Type::Bytes,
            ValueTy::Option(v) => mir::Type::Option(Box::new(Self::lower_value_type(v))),
        }
    }

    fn lower_expr(&self, expr: &hir::expression::Expression) -> mir::Expression {
        let ty = Self::lower_value_type(self.mode.expr_type(expr.eid).get_value_type());
        mir::Expression {
            kind: self.lower_expression_kind(&expr.kind, &ty),
            ty,
        }
    }

    fn lower_expression_kind(
        &self,
        expr: &hir::expression::ExpressionKind,
        ty: &mir::Type,
    ) -> mir::ExpressionKind {
        match expr {
            hir::expression::ExpressionKind::LoadConstant(constant) => {
                mir::ExpressionKind::LoadConstant(Self::lower_constant(constant, ty))
            }
            hir::expression::ExpressionKind::ArithLog(op, args) => {
                let op = Self::lower_arith_log_op(*op);
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::ArithLog(op, args)
            }
            hir::expression::ExpressionKind::StreamAccess(sr, kind, para) => {
                mir::ExpressionKind::StreamAccess(
                    *sr,
                    *kind,
                    para.iter().map(|p| self.lower_expr(p)).collect(),
                )
            }
            hir::expression::ExpressionKind::ParameterAccess(_sr, _para) => unimplemented!(),
            hir::expression::ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let condition = Box::new(self.lower_expr(condition));
                let consequence = Box::new(self.lower_expr(consequence));
                let alternative = Box::new(self.lower_expr(alternative));
                mir::ExpressionKind::Ite {
                    condition,
                    consequence,
                    alternative,
                }
            }
            hir::expression::ExpressionKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| self.lower_expr(element))
                    .collect::<Vec<mir::Expression>>();
                mir::ExpressionKind::Tuple(elements)
            }
            hir::expression::ExpressionKind::TupleAccess(tuple, element_pos) => {
                let tuple = Box::new(self.lower_expr(tuple));
                let element_pos = *element_pos;
                mir::ExpressionKind::TupleAccess(tuple, element_pos)
            }
            hir::expression::ExpressionKind::Function {
                name,
                args,
                type_param: _,
            } => {
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<Vec<mir::Expression>>();
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

    fn lower_constant(constant: &hir::expression::Constant, ty: &mir::Type) -> mir::Constant {
        match constant {
            hir::expression::Constant::BasicConstant(lit) => Self::lower_constant_literal(lit, ty),
            hir::expression::Constant::InlinedConstant(lit, _ty) => {
                Self::lower_constant_literal(lit, ty)
            }
        }
    }

    fn lower_constant_literal(
        constant: &hir::expression::ConstantLiteral,
        ty: &mir::Type,
    ) -> mir::Constant {
        match constant {
            hir::expression::ConstantLiteral::Str(s) => mir::Constant::Str(s.clone()),
            hir::expression::ConstantLiteral::Bool(b) => mir::Constant::Bool(*b),
            hir::expression::ConstantLiteral::Integer(i) => match ty {
                mir::Type::Int(_) => mir::Constant::Int(*i),
                mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                _ => unreachable!(),
            },
            hir::expression::ConstantLiteral::SInt(i) => {
                //TODO rewrite to 128 bytes
                match ty {
                    mir::Type::Int(_) => mir::Constant::Int(*i as i64),
                    mir::Type::UInt(_) => mir::Constant::UInt(*i as u64),
                    _ => unreachable!(),
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::IrExprMode;
    use crate::parse::parse;
    use crate::FrontendConfig;
    use rtlola_reporting::Handler;
    use std::path::PathBuf;
    fn lower_spec(spec: &str) -> (hir::RTLolaHIR<CompleteMode>, mir::RTLolaMIR) {
        let handler = Handler::new(PathBuf::new(), spec.into());
        let config = FrontendConfig::default();
        let ast = parse(spec, &handler, config).unwrap_or_else(|e| panic!("{}", e));
        let hir = Hir::<IrExprMode>::from_ast(ast, &handler, &config)
            .unwrap()
            .build_dependency_graph()
            .unwrap()
            .type_check(&handler)
            .unwrap()
            .build_evaluation_order()
            .compute_memory_bounds()
            .finalize();
        (hir.clone(), hir.lower())
    }

    #[test]
    fn check_event_based_streams() {
        let spec = "input a: Float32\ninput b:Float32\noutput c := a + b\noutput d := a.hold().defaults(to:0.0) + b\noutput e := a + 9.0\ntrigger d < e\ntrigger a < 5.0";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 2);
        assert_eq!(mir.outputs.len(), 5);
        assert_eq!(mir.event_driven.len(), 5);
        assert_eq!(mir.time_driven.len(), 0);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 0);
        assert_eq!(mir.triggers.len(), 2);
        let hir_a = hir.inputs().find(|i| i.name == "a".to_string()).unwrap();
        let mir_a = mir
            .inputs
            .iter()
            .find(|i| i.name == "a".to_string())
            .unwrap();
        assert_eq!(hir_a.sr, mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir
            .outputs
            .iter()
            .find(|i| i.name == "d".to_string())
            .unwrap();
        assert_eq!(hir_d.sr, mir_d.reference);
    }

    #[test]
    fn check_time_driven_streams() {
        let spec = "input a: Int8\ninput b:Int8\noutput c @1Hz:= a.aggregate(over: 2s, using: sum)+ b.aggregate(over: 4s, using:sum)\noutput d @4Hz:= a.hold().defaults(to:0) + b.hold().defaults(to: 0)\noutput e @0.5Hz := a.aggregate(over: 4s, using: sum) + 9\ntrigger d < e";
        let (hir, mir) = lower_spec(spec);

        assert_eq!(mir.inputs.len(), 2);
        assert_eq!(mir.outputs.len(), 4);
        assert_eq!(mir.event_driven.len(), 0);
        assert_eq!(mir.time_driven.len(), 4);
        assert_eq!(mir.discrete_windows.len(), 0);
        assert_eq!(mir.sliding_windows.len(), 3);
        assert_eq!(mir.triggers.len(), 1);
        let hir_a = hir.inputs().find(|i| i.name == "a".to_string()).unwrap();
        let mir_a = mir
            .inputs
            .iter()
            .find(|i| i.name == "a".to_string())
            .unwrap();
        assert_eq!(hir_a.sr, mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir
            .outputs
            .iter()
            .find(|i| i.name == "d".to_string())
            .unwrap();
        assert_eq!(hir_d.sr, mir_d.reference);
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
        let mir_a = mir
            .inputs
            .iter()
            .find(|i| i.name == "a".to_string())
            .unwrap();
        assert_eq!(hir_a.sr, mir_a.reference);
        let hir_d = hir.outputs().find(|i| i.name == "d".to_string()).unwrap();
        let mir_d = mir
            .outputs
            .iter()
            .find(|i| i.name == "d".to_string())
            .unwrap();
        assert_eq!(hir_d.sr, mir_d.reference);
    }
}
