use super::*;
extern crate regex;

use crate::common_ir::{Offset, StreamAccessKind, StreamReference};
use crate::hir::expression::{Constant, ConstantLiteral, ExprId, Expression, ExpressionKind, ValueEq};
use crate::hir::modes::HirMode;
use crate::hir::modes::IrExprTrait;
use crate::hir::{Ac, Input, Output, SpawnTemplate, Trigger};
use crate::reporting::Span;
use crate::tyc::pacing_types::{
    AbstractExpressionType, AbstractPacingType, ActivationCondition, ConcretePacingType, ConcreteStreamPacing, Freq,
    InferredTemplates, PacingErrorKind, StreamTypeKeys,
};
use crate::tyc::rtltc::{NodeId, TypeError};
use crate::RTLolaHIR;
use rusttyc::TypeTable;
use rusttyc::{TcKey, TypeChecker};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable(String);

impl rusttyc::TcVar for Variable {}

pub struct PacingTypeChecker<'a, M>
where
    M: HirMode + IrExprTrait + 'static,
{
    pub(crate) hir: &'a RTLolaHIR<M>,
    pub(crate) pacing_tyc: TypeChecker<AbstractPacingType, Variable>,
    pub(crate) expression_tyc: TypeChecker<AbstractExpressionType, Variable>,
    pub(crate) node_key: HashMap<NodeId, StreamTypeKeys>,
    pub(crate) pacing_key_span: HashMap<TcKey, Span>,
    pub(crate) expression_key_span: HashMap<TcKey, Span>,
    pub(crate) names: &'a HashMap<StreamReference, &'a str>,
    pub(crate) annotated_checks: HashMap<TcKey, (ConcretePacingType, TcKey)>,
}

impl<'a, M> PacingTypeChecker<'a, M>
where
    M: HirMode + IrExprTrait + 'static,
{
    pub(crate) fn new(hir: &'a RTLolaHIR<M>, names: &'a HashMap<StreamReference, &'a str>) -> Self {
        let node_key = HashMap::new();
        let pacing_tyc = TypeChecker::new();
        let expression_tyc = TypeChecker::new();
        let pacing_key_span = HashMap::new();
        let expression_key_span = HashMap::new();
        let annotated_checks = HashMap::new();
        let mut res = PacingTypeChecker {
            hir,
            pacing_tyc,
            expression_tyc,
            node_key,
            pacing_key_span,
            expression_key_span,
            names,
            annotated_checks,
        };
        res.generate_keys_for_streams();
        res
    }

    pub(crate) fn new_stream_key(&mut self) -> StreamTypeKeys {
        let close = self.expression_tyc.new_term_key();
        self.expression_tyc
            .impose(close.concretizes_explicit(AbstractExpressionType::AnyClose))
            .expect("close key cannot be bound otherwise yet");
        StreamTypeKeys {
            exp_pacing: self.pacing_tyc.new_term_key(),
            spawn: (self.pacing_tyc.new_term_key(), self.expression_tyc.new_term_key()),
            filter: self.expression_tyc.new_term_key(),
            close,
        }
    }

    pub(crate) fn add_span_to_stream_key(&mut self, keys: StreamTypeKeys, span: Span) {
        self.pacing_key_span.insert(keys.exp_pacing, span.clone());
        self.pacing_key_span.insert(keys.spawn.0, span.clone());
        self.expression_key_span.insert(keys.spawn.1, span.clone());
        self.expression_key_span.insert(keys.filter, span.clone());
        self.expression_key_span.insert(keys.close, span);
    }

    pub(crate) fn impose_more_concrete(
        &mut self,
        keys_l: StreamTypeKeys,
        keys_r: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        self.pacing_tyc.impose(keys_l.exp_pacing.concretizes(keys_r.exp_pacing))?;
        self.pacing_tyc.impose(keys_l.spawn.0.concretizes(keys_r.spawn.0))?;
        self.expression_tyc.impose(keys_l.spawn.1.concretizes(keys_r.spawn.1))?;
        self.expression_tyc.impose(keys_l.filter.concretizes(keys_r.filter))?;
        self.expression_tyc.impose(keys_l.close.concretizes(keys_r.close))?;
        Ok(())
    }

    pub(crate) fn generate_keys_for_streams(&mut self) {
        for input in self.hir.inputs() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(input.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, input.span.clone());
            self.pacing_key_span.insert(key.spawn.0, Span::Unknown);
            self.expression_key_span.insert(key.spawn.1, Span::Unknown);
            self.expression_key_span.insert(key.filter, Span::Unknown);
            self.expression_key_span.insert(key.close, Span::Unknown);
        }
        for output in self.hir.outputs() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(output.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, self.hir.expr(output.sr).span.clone());
            self.pacing_key_span.insert(
                key.spawn.0,
                output
                    .instance_template
                    .spawn
                    .as_ref()
                    .and_then(|spawn| spawn.target)
                    .map(|id| self.hir.expression(id).span.clone())
                    .unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.spawn.1,
                output
                    .instance_template
                    .spawn
                    .as_ref()
                    .and_then(|spawn| spawn.condition)
                    .map(|id| self.hir.expression(id).span.clone())
                    .unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.filter,
                output.instance_template.filter.map(|id| self.hir.expression(id).span.clone()).unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.close,
                output.instance_template.close.map(|id| self.hir.expression(id).span.clone()).unwrap_or(Span::Unknown),
            );

            // Create Stream Parameters
            for (idx, parameter) in output.params.iter().enumerate() {
                let key = self.new_stream_key();
                self.node_key.insert(NodeId::Param(idx, output.sr), key);
                self.add_span_to_stream_key(key, parameter.span.clone());
            }
        }

        for trigger in self.hir.triggers() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(trigger.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, trigger.span.clone());
            self.pacing_key_span.insert(key.spawn.0, Span::Unknown);
            self.expression_key_span.insert(key.spawn.1, Span::Unknown);
            self.expression_key_span.insert(key.filter, Span::Unknown);
            self.expression_key_span.insert(key.close, Span::Unknown);
        }
    }

    /// Binds the key to the given annotated type
    pub(crate) fn bind_to_annotated_type(
        &mut self,
        target: TcKey,
        bound: &Ac,
        conflict_key: TcKey,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let concrete_pacing = ConcretePacingType::from_ac(bound, self.hir)?;
        self.annotated_checks.insert(target, (concrete_pacing, conflict_key));
        Ok(())
    }

    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<(), TypeError<PacingErrorKind>> {
        let ac = AbstractPacingType::Event(ActivationCondition::Stream(input.sr));
        let keys = self.node_key[&NodeId::SRef(input.sr)];
        self.pacing_tyc.impose(keys.exp_pacing.concretizes_explicit(ac))?;
        Ok(())
    }

    pub(crate) fn trigger_infer(&mut self, trigger: &Trigger) -> Result<(), TypeError<PacingErrorKind>> {
        let ex_key = self.expression_infer(self.hir.expr(trigger.sr))?;
        //Todo: add explicit pacing annotation
        let trigger_key = self.node_key[&NodeId::SRef(trigger.sr)].exp_pacing;
        self.pacing_tyc.impose(trigger_key.equate_with(ex_key.exp_pacing))?;
        Ok(())
    }

    pub(crate) fn output_infer(&mut self, output: &Output) -> Result<(), TypeError<PacingErrorKind>> {
        let stream_keys = self.node_key[&NodeId::SRef(output.sr)];

        // Type Expression Pacing
        let exp_key = self.expression_infer(&self.hir.expr(output.sr))?;

        // Check if there is a type is annotated
        if let Some(ac) = &output.activation_condition {
            let (annotated_ty, span) = AbstractPacingType::from_ac(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.exp_pacing, span);

            self.bind_to_annotated_type(stream_keys.exp_pacing, ac, exp_key.exp_pacing)?;
            self.pacing_tyc.impose(stream_keys.exp_pacing.concretizes_explicit(annotated_ty))?;
            self.pacing_tyc.impose(stream_keys.exp_pacing.concretizes(exp_key.exp_pacing))?;
        } else {
            // Output type is equal to inferred type
            self.pacing_tyc.impose(stream_keys.exp_pacing.equate_with(exp_key.exp_pacing))?;
        }

        // Type spawn condition
        if let Some(spawn) = output.instance_template.spawn.as_ref() {
            self.spawn_infer(spawn, stream_keys, exp_key)?;
        }

        // Type filter
        if let Some(exp_id) = output.instance_template.filter {
            self.filter_infer(exp_id, stream_keys, exp_key)?;
        }

        //Type close
        if let Some(exp_id) = output.instance_template.close {
            self.close_infer(exp_id, stream_keys, exp_key)?;
        }
        Ok(())
    }

    pub(crate) fn spawn_infer(
        &mut self,
        spawn: &SpawnTemplate,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let spawn_target_keys = self.new_stream_key();
        let spawn_condition_keys = self.new_stream_key();

        // Check if there is a pacing annotated
        if let Some(ac) = spawn.pacing.as_ref() {
            let (annotated_ty, span) = AbstractPacingType::from_ac(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.spawn.0, span);
            self.pacing_tyc.impose(stream_keys.spawn.0.concretizes_explicit(annotated_ty))?;
            self.bind_to_annotated_type(stream_keys.spawn.0, ac, spawn_target_keys.exp_pacing)?;
        }

        // Type spawn target
        let spawn_exp = spawn.target.map(|eid| self.hir.expression(eid));
        if let Some(spawn_target) = spawn_exp {
            let inferred = self.expression_infer(spawn_target)?;
            self.node_key.insert(NodeId::Expr(spawn_target.eid), spawn_target_keys);
            self.add_span_to_stream_key(spawn_target_keys, spawn_target.span.clone());
            self.impose_more_concrete(spawn_target_keys, inferred)?;
        }

        // Type spawn condition
        let spawn_condition_exp = spawn.condition.map(|eid| self.hir.expression(eid).clone());
        if let Some(spawn_condition) = spawn_condition_exp {
            self.node_key.insert(NodeId::Expr(spawn_condition.eid), spawn_condition_keys);
            self.add_span_to_stream_key(spawn_condition_keys, spawn_condition.span.clone());
            let inferred = self.expression_infer(&spawn_condition)?;
            self.impose_more_concrete(spawn_condition_keys, inferred)?;

            //Streams spawn condition is equal to annotated condition
            self.expression_tyc.impose(
                stream_keys.spawn.1.concretizes_explicit(AbstractExpressionType::Expression(spawn_condition)),
            )?;
        }

        // Pacing of spawn target is more concrete than pacing of condition
        self.pacing_tyc.impose(spawn_target_keys.exp_pacing.concretizes(spawn_condition_keys.exp_pacing))?;
        // Spawn condition is more concrete than the spawn condition of the expression
        self.expression_tyc.impose(stream_keys.spawn.1.concretizes(exp_keys.spawn.1))?;
        // Spawn target is more concrete than the spawn target of the expression
        self.pacing_tyc.impose(stream_keys.spawn.0.concretizes(exp_keys.spawn.0))?;
        // Spawn pacing of the stream is more concrete than the spawn pacing of the target
        self.pacing_tyc.impose(stream_keys.spawn.0.concretizes(spawn_target_keys.exp_pacing))?;

        Ok(())
    }

    pub(crate) fn filter_infer(
        &mut self,
        filter_id: ExprId,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let filter = self.hir.expression(filter_id);
        let filter_keys = self.expression_infer(filter)?;

        //Pacing of stream is more concrete than pacing of the filter
        self.pacing_tyc.impose(stream_keys.exp_pacing.concretizes(filter_keys.exp_pacing))?;
        //Filter is equal to the expression
        self.expression_tyc
            .impose(stream_keys.filter.concretizes_explicit(AbstractExpressionType::Expression(filter.clone())))?;
        //Filter of the stream is more concrete than the filter of the streams expression
        self.expression_tyc.impose(stream_keys.filter.concretizes(exp_keys.filter))?;
        Ok(())
    }

    pub(crate) fn close_infer(
        &mut self,
        close_id: ExprId,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let close = self.hir.expression(close_id);

        // We have no constraints on the type of the close expression
        self.expression_infer(close)?;

        //Close is equal to the expression
        self.expression_tyc
            .impose(stream_keys.close.concretizes_explicit(AbstractExpressionType::Expression(close.clone())))?;
        //Close of the streams expression is more concrete than the close of the stream
        self.expression_tyc.impose(exp_keys.close.concretizes(stream_keys.close))?;
        Ok(())
    }

    fn handle_offset(
        &mut self,
        kind: &StreamAccessKind,
        term_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        if let StreamAccessKind::Offset(off) = kind {
            match off {
                Offset::PastRealTime(_) | Offset::FutureRealTime(_) => {
                    // Real time offset are only allowed on timed streams.
                    debug_assert!(false, "Real-Time offsets not supported");
                    self.pacing_tyc
                        .impose(term_keys.exp_pacing.concretizes_explicit(AbstractPacingType::Periodic(Freq::Any)))?;
                }
                Offset::PastDiscrete(_) | Offset::FutureDiscrete(_) => {}
            }
        }
        Ok(())
    }

    pub(crate) fn expression_infer(&mut self, exp: &Expression) -> Result<StreamTypeKeys, TypeError<PacingErrorKind>> {
        let term_keys: StreamTypeKeys = self.new_stream_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::LoadConstant(_) | ExpressionKind::ParameterAccess(_, _) => {
                //constants have arbitrary pacing type
            }
            ExpressionKind::StreamAccess(sref, kind, args) => {
                let stream_key = self.node_key[&NodeId::SRef(*sref)];
                match kind {
                    StreamAccessKind::Sync | StreamAccessKind::Offset(_) => {
                        self.handle_offset(kind, term_keys)?;
                        self.impose_more_concrete(term_keys, stream_key)?;

                        //Check that arguments are equal to spawn target if parameterized
                        let target = self.hir.spawn(*sref).and_then(|(t, _)| t).map(|target| match &target.kind {
                            ExpressionKind::Tuple(s) => s.clone(),
                            _ => vec![target.clone()],
                        });

                        if let Some(spawn_args) = target {
                            if spawn_args.len() != args.len()
                                || spawn_args.iter().zip(args.iter()).any(|(l, r)| l.value_neq(r))
                            {
                                return Err(PacingErrorKind::Other(
                                    exp.span.clone(),
                                    format!(
                                        "Expected spawn arguments: ({}) but found: ({})",
                                        spawn_args
                                            .iter()
                                            .map(|e| e.pretty_string(&self.names))
                                            .collect::<Vec<String>>()
                                            .join(", "),
                                        args.iter()
                                            .map(|e| e.pretty_string(&self.names))
                                            .collect::<Vec<String>>()
                                            .join(", ")
                                    ),
                                    vec![],
                                )
                                .into());
                            }
                        }
                    }
                    StreamAccessKind::Hold => {}
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => {
                        self.pacing_tyc.impose(term_keys.exp_pacing.concretizes_explicit(Periodic(Freq::Any)))?;
                        // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
                    }
                };

                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            }
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr)?;
                let def_key = self.expression_infer(&*default)?;

                self.impose_more_concrete(term_keys, ex_key)?;
                self.impose_more_concrete(term_keys, def_key)?;
            }
            ExpressionKind::ArithLog(_, args) => match args.len() {
                2 => {
                    let left_key = self.expression_infer(&args[0])?;
                    let right_key = self.expression_infer(&args[1])?;

                    self.impose_more_concrete(term_keys, left_key)?;
                    self.impose_more_concrete(term_keys, right_key)?;
                }
                1 => {
                    let ex_key = self.expression_infer(&args[0])?;
                    self.impose_more_concrete(term_keys, ex_key)?;
                }
                _ => unreachable!(),
            },
            ExpressionKind::Ite { condition, consequence, alternative } => {
                let cond_key = self.expression_infer(&*condition)?;
                let cons_key = self.expression_infer(&*consequence)?;
                let alt_key = self.expression_infer(&*alternative)?;

                self.impose_more_concrete(term_keys, cond_key)?;
                self.impose_more_concrete(term_keys, cons_key)?;
                self.impose_more_concrete(term_keys, alt_key)?;
            }
            ExpressionKind::Tuple(elements) => {
                for e in elements {
                    let ele_keys = self.expression_infer(e)?;
                    self.impose_more_concrete(term_keys, ele_keys)?;
                }
            }
            ExpressionKind::Function { args, .. } => {
                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            }
            ExpressionKind::TupleAccess(t, _) => {
                let exp_key = self.expression_infer(&*t)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            }
            ExpressionKind::Widen(inner, _) => {
                let exp_key = self.expression_infer(&*inner)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            }
        };
        self.node_key.insert(NodeId::Expr(exp.eid), term_keys);
        self.add_span_to_stream_key(term_keys, exp.span.clone());
        Ok(term_keys)
    }

    fn check_explicit_bounds(
        checks: HashMap<TcKey, (ConcretePacingType, TcKey)>,
        tt: &TypeTable<AbstractPacingType>,
    ) -> Vec<TypeError<PacingErrorKind>> {
        checks
            .into_iter()
            .filter_map(|(key, (bound, conflict_key))| {
                let inferred = tt[&key].clone();
                if inferred != bound {
                    Some(TypeError {
                        kind: PacingErrorKind::PacingTypeMismatch(bound, inferred),
                        key1: Some(key),
                        key2: Some(conflict_key),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn is_parameterized(
        keys: StreamTypeKeys,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &TypeTable<AbstractExpressionType>,
    ) -> bool {
        let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
        let spawn_cond = exp_tt[&keys.spawn.1].clone();
        let filter = exp_tt[&keys.filter].clone();
        let close = exp_tt[&keys.close].clone();

        let kind_true = ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true)));
        let kind_false = ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false)));

        spawn_pacing != ConcretePacingType::Constant
            || spawn_cond.kind.value_neq(&kind_true)
            || filter.kind.value_neq(&kind_true)
            || close.kind.value_neq(&kind_false)
    }

    pub(crate) fn post_process(
        hir: &RTLolaHIR<M>,
        nid_key: HashMap<NodeId, StreamTypeKeys>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &TypeTable<AbstractExpressionType>,
    ) -> Vec<TypeError<PacingErrorKind>> {
        let mut errors = vec![];

        // Check that every periodic stream has a frequency
        for output in hir.outputs() {
            let ct = &pacing_tt[&nid_key[&NodeId::SRef(output.sr)].exp_pacing];
            match ct {
                ConcretePacingType::Periodic => {
                    errors.push(PacingErrorKind::FreqAnnotationNeeded(output.span.clone()).into());
                }
                ConcretePacingType::Constant => {
                    errors.push(PacingErrorKind::NeverEval(output.span.clone()).into());
                }
                _ => {}
            }
        }

        //Check that trigger expression does not access parameterized stream
        for trigger in hir.triggers() {
            let keys = nid_key[&NodeId::Expr(trigger.expr_id)];
            if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                errors.push(
                    PacingErrorKind::ParameterizationNotAllowed(hir.expression(trigger.expr_id).span.clone()).into(),
                );
            }
        }

        //Check that spawn target/condition, filter, close is not again parameterized
        for output in hir.outputs() {
            if let Some(template) = output.instance_template.spawn.as_ref() {
                //Spawn target
                if let Some(target) = template.target {
                    let keys = nid_key[&NodeId::Expr(target)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                        errors.push(
                            PacingErrorKind::ParameterizationNotAllowed(hir.expression(target).span.clone()).into(),
                        );
                    }
                }
                //Spawn condition
                if let Some(condition) = template.condition {
                    let keys = nid_key[&NodeId::Expr(condition)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                        errors.push(
                            PacingErrorKind::ParameterizationNotAllowed(hir.expression(condition).span.clone()).into(),
                        );
                    }
                }
            }
            //Filter
            if let Some(filter) = output.instance_template.filter {
                let keys = nid_key[&NodeId::Expr(filter)];
                if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                    errors
                        .push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(filter).span.clone()).into());
                }
            }
            //Close
            if let Some(close) = output.instance_template.close {
                let keys = nid_key[&NodeId::Expr(close)];
                if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                    errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(close).span.clone()).into());
                }
            }
        }

        //Check that spawn pacing is not constant
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::SRef(output.sr)];
            let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
            if let Some(template) = output.instance_template.spawn.as_ref() {
                if spawn_pacing == ConcretePacingType::Constant {
                    let span = template
                        .pacing
                        .as_ref()
                        .map(|ac| match ac {
                            Ac::Frequency { span, .. } => span.clone(),
                            Ac::Expr(id) => hir.expression(*id).span.clone(),
                        })
                        .or_else(|| template.target.map(|id| hir.expression(id).span.clone()))
                        .or_else(|| template.condition.map(|id| hir.expression(id).span.clone()))
                        .unwrap_or_else(|| output.span.clone());
                    errors.push(
                        PacingErrorKind::Other(
                            span,
                            "No instance is created as spawn pacing is 'Constant'".into(),
                            vec![],
                        )
                        .into(),
                    )
                }
            }
        }
        //Check that stream without spawn template does not access parameterized stream
        //Check that stream without filter does not access filtered stream
        //Check that stream without close does not access closed stream
        let kind_true = ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true)));
        let kind_false = ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false)));
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::Expr(output.expr_id)];
            let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
            let spawn_cond = exp_tt[&keys.spawn.1].clone();
            let filter_type = exp_tt[&keys.filter].clone();
            let close_type = exp_tt[&keys.close].clone();

            let spawn = if output.instance_template.spawn.is_none()
                && (spawn_pacing != ConcretePacingType::Constant || spawn_cond.kind.value_neq(&kind_true))
            {
                Some((spawn_pacing, spawn_cond))
            } else {
                None
            };
            let filter = if output.instance_template.filter.is_none() && filter_type.kind.value_neq(&kind_true) {
                Some(filter_type)
            } else {
                None
            };
            let close = if output.instance_template.close.is_none() && close_type.kind.value_neq(&kind_false) {
                Some(close_type)
            } else {
                None
            };

            if spawn.is_some() || filter.is_some() || close.is_some() {
                errors.push(
                    PacingErrorKind::ParameterizationNeeded {
                        who: output.span.clone(),
                        why: hir.expression(output.expr_id).span.clone(),
                        inferred: Box::new(InferredTemplates { spawn, filter, close }),
                    }
                    .into(),
                )
            }
        }

        //Check that every output that has parameters has a spawn condition with a target and the other way around
        for output in hir.outputs() {
            if !output.params.is_empty() && output.instance_template.spawn.as_ref().and_then(|s| s.target).is_none() {
                errors.push(
                    PacingErrorKind::Other(
                        output.span.clone(),
                        "A spawn declaration is needed to initialize the parameters of the stream.".into(),
                        vec![],
                    )
                    .into(),
                );
            }
            if let Some(target) = output.instance_template.spawn.as_ref().and_then(|s| s.target) {
                if output.params.is_empty() {
                    let span = hir.expression(target).span.clone();
                    errors.push(
                        PacingErrorKind::Other(
                            span,
                            "Found a spawn target declaration in a stream without parameter.".into(),
                            vec![],
                        )
                        .into(),
                    );
                }
            }
        }

        //Warning unintuitive spawn type
        for output in hir.outputs() {
            if let Some(template) = output.instance_template.spawn.as_ref() {
                if let Some(target_id) = template.target {
                    let target_type = pacing_tt[&nid_key[&NodeId::Expr(target_id)].exp_pacing].clone();
                    let spawn_pacing = pacing_tt[&nid_key[&NodeId::SRef(output.sr)].spawn.0].clone();
                    if template.pacing.is_none() && target_type != spawn_pacing {
                        errors.push(
                            PacingErrorKind::UnintuitivePacingWarning(
                                hir.expression(target_id).span.clone(),
                                spawn_pacing,
                            )
                            .into(),
                        );
                    }
                }
            }
        }

        //Warning unintuitive exp pacing
        for output in hir.outputs() {
            let exp_pacing = pacing_tt[&nid_key[&NodeId::Expr(output.expr_id)].exp_pacing].clone();
            let stream_pacing = pacing_tt[&nid_key[&NodeId::SRef(output.sr)].exp_pacing].clone();
            if output.activation_condition.is_none() && exp_pacing != stream_pacing {
                errors.push(PacingErrorKind::UnintuitivePacingWarning(output.span.clone(), stream_pacing).into());
            }
        }

        //Check Periodic Stream with Spawn accesses on periodic streams
        for output in hir.outputs() {
            let stream_keys = nid_key[&NodeId::SRef(output.sr)];
            let exp_pacing = pacing_tt[&stream_keys.exp_pacing].clone();
            let spawn_pacing = pacing_tt[&stream_keys.spawn.0].clone();
            let spawn_cond = exp_tt[&stream_keys.spawn.1].clone();
            if matches!(exp_pacing, ConcretePacingType::FixedPeriodic(_))
                && spawn_pacing != ConcretePacingType::Constant
            {
                let accesses_streams = hir.expr(output.sr).get_sync_accesses();
                for target in accesses_streams {
                    let target_keys = nid_key[&NodeId::SRef(target)];
                    let target_spawn_pacing = pacing_tt[&target_keys.spawn.0].clone();
                    let target_spawn_condition = exp_tt[&target_keys.spawn.1].clone();
                    if spawn_pacing != target_spawn_pacing || spawn_cond.value_neq(&target_spawn_condition) {
                        let target_span =
                            hir.outputs().find(|o| o.sr == target).map(|o| o.span.clone()).unwrap_or(Span::Unknown);
                        errors.push(
                            PacingErrorKind::SpawnPeriodicMismatch(
                                output.span.clone(),
                                target_span,
                                (spawn_pacing.clone(), spawn_cond.clone()),
                            )
                            .into(),
                        );
                    }
                }
            }
        }

        errors
    }

    pub(crate) fn type_check(mut self, handler: &Handler) -> Option<HashMap<NodeId, ConcreteStreamPacing>> {
        for input in self.hir.inputs() {
            if let Err(e) = self.input_infer(input) {
                e.emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
            }
        }

        for output in self.hir.outputs() {
            if let Err(e) = self.output_infer(output) {
                e.emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
            }
        }

        for trigger in self.hir.triggers() {
            if let Err(e) = self.trigger_infer(trigger) {
                e.emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
            }
        }

        let nid_key = self.node_key.clone();
        let pacing_tt = match self.pacing_tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                TypeError::from(e).emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
                return None;
            }
        };
        let exp_tt = match self.expression_tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                TypeError::from(e).emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
                return None;
            }
        };

        if handler.contains_error() {
            return None;
        }

        for pe in Self::check_explicit_bounds(self.annotated_checks, &pacing_tt) {
            pe.emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
        }
        if handler.contains_error() {
            return None;
        }

        for pe in Self::post_process(&self.hir, nid_key, &pacing_tt, &exp_tt) {
            pe.emit(handler, &[&self.pacing_key_span, &self.expression_key_span], &self.names);
        }
        if handler.contains_error() {
            return None;
        }

        let ctt: HashMap<NodeId, ConcreteStreamPacing> = self
            .node_key
            .iter()
            .map(|(id, key)| {
                let exp_pacing = pacing_tt[&key.exp_pacing].clone();
                let spawn_pacing = pacing_tt[&key.spawn.0].clone();
                let spawn_condition_expression = exp_tt[&key.spawn.1].clone();
                let filter = exp_tt[&key.filter].clone();
                let close = exp_tt[&key.close].clone();

                (
                    *id,
                    ConcreteStreamPacing {
                        expression_pacing: exp_pacing,
                        spawn: (spawn_pacing, spawn_condition_expression),
                        filter,
                        close,
                    },
                )
            })
            .collect();

        Some(ctt)
    }
}

#[cfg(test)]
mod tests {
    use crate::common_ir::{StreamAccessKind, StreamReference};
    use crate::hir::expression::{ArithLogOp, Constant, ConstantLiteral, ExprId, Expression, ExpressionKind, ValueEq};
    use crate::hir::modes::IrExprMode;
    use crate::hir::RTLolaHIR;
    use crate::reporting::{Handler, Span};
    use crate::tyc::pacing_types::{ActivationCondition, ConcretePacingType};
    use crate::tyc::rtltc::NodeId;
    use crate::tyc::LolaTypeChecker;
    use crate::RTLolaAst;
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use std::path::PathBuf;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    macro_rules! assert_value_eq {
        ($left:expr, $right:expr) => {{
            match (&$left, &$right) {
                (left_val, right_val) => {
                    if (left_val.value_neq(right_val)) {
                        // The reborrows below are intentional. Without them, the stack slot for the
                        // borrow is initialized even before the values are compared, leading to a
                        // noticeable slow down.
                        panic!(
                            r#"assertion failed: `left.value_eq(right)`
      left: `{:?}`,
     right: `{:?}`"#,
                            &*left_val, &*right_val
                        )
                    }
                }
            }
        }};
    }

    fn setup_ast(spec: &str) -> (RTLolaHIR<IrExprMode>, Handler) {
        let handler = crate::reporting::Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst = match crate::parse::parse(spec, &handler, crate::FrontendConfig::default()) {
            Ok(s) => s,
            Err(e) => panic!("Spec {} cannot be parsed: {}", spec, e),
        };
        let hir = crate::hir::RTLolaHIR::<IrExprMode>::from_ast(ast, &handler, &crate::FrontendConfig::default());
        (hir, handler)
    }

    fn num_errors(spec: &str) -> usize {
        let (spec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&spec, &handler);
        ltc.pacing_type_infer();
        return handler.emitted_errors();
    }

    fn get_sr_for_name(hir: &RTLolaHIR<IrExprMode>, name: &str) -> StreamReference {
        if let Some(i) = hir.get_input_with_name(name) {
            i.sr
        } else {
            hir.get_output_with_name(name).unwrap().sr
        }
    }
    fn get_node_for_name(hir: &RTLolaHIR<IrExprMode>, name: &str) -> NodeId {
        NodeId::SRef(get_sr_for_name(hir, name))
    }

    #[test]
    fn test_input_simple() {
        let spec = "input i: Int8";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "i")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_output_simple() {
        let spec = "input a: Int8\n input b: Int8 \n output o := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        let ac_b = ActivationCondition::Stream(get_sr_for_name(&hir, "b"));
        assert_eq!(
            tt[&get_node_for_name(&hir, "o")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_trigger_simple() {
        let spec = "input a: Int8\n trigger a == 42";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        assert_eq!(
            tt[&NodeId::SRef(hir.triggers().nth(0).unwrap().sr)].expression_pacing,
            ConcretePacingType::Event(ac_a)
        );
    }

    #[test]
    fn test_disjunction_annotated() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        let ac_b = ActivationCondition::Stream(get_sr_for_name(&hir, "b"));
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "x")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Disjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_frequency_simple() {
        let spec = "output a: UInt8 @10Hz := 0";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "a")].expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(10).unwrap()))
        );
    }

    #[test]
    fn test_frequency_conjunction() {
        let spec = "output a: Int32 @10Hz := 0\noutput b: Int32 @5Hz := 0\noutput x := a+b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "x")].expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap()))
        );
    }

    #[test]
    fn test_get_not_possible() {
        // it should not be possible to use get with RealTime and EventBased streams
        let spec = "input a: Int32\noutput x: Int32 @ 1Hz := 0\noutput y:Int32 @ a := x.get().defaults(to: 0)";
        assert_eq!(num_errors(spec), 1);
        let spec = "input a: Int32\noutput x: Int32 @ a := 0\noutput y:Int32 @ 1Hz := x.get().defaults(to: 0)";
        assert_eq!(num_errors(spec), 1);
    }

    #[test]
    fn test_normalization_event_streams() {
        let spec = "input a: Int32\ninput b: Int32\ninput c: Int32\noutput x := a + b\noutput y := x + x + c";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        //  input `a` has NodeId =  1
        let a_id = get_sr_for_name(&hir, "a");
        //  input `b` has NodeId =  3
        let b_id = get_sr_for_name(&hir, "b");
        //  input `c` has NodeId =  5
        let c_id = get_sr_for_name(&hir, "c");
        // output `x` has NodeId = 11
        let x_id = get_node_for_name(&hir, "x");
        // output `y` has NodeId = 19
        let y_id = get_node_for_name(&hir, "y");

        assert_eq!(
            tt[&x_id].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(a_id),
                ActivationCondition::Stream(b_id)
            ]))
        );

        assert_eq!(
            tt[&y_id].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(a_id),
                ActivationCondition::Stream(b_id),
                ActivationCondition::Stream(c_id)
            ]))
        );
    }

    #[test]
    fn test_output_in_ac() {
        let spec = "input a: Int32\ninput b: Int32\ninput c: Int32\noutput x := a + b\noutput y @(x & c) := a + b + c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_counter() {
        let spec = "output b := b.offset(by: -1).defaults(to: 0) + 1";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset() {
        let spec = "output b @2Hz := b[-1].defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_faster() {
        let spec = "output a @4Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // equivalent to b[-500ms].defaults(to: 0) + a[-250ms].defaults(to: 0)
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_incompatible() {
        let spec = "output a @3Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_simple_loop() {
        let spec = "output a @1Hz := a";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_sample_and_hold() {
        let spec = "
            output a @3Hz := 0
            output a_offset := a[-1].defaults(to: 0)
            output b @2Hz := b[-1].defaults(to: 0) + a_offset.hold().defaults(to: 0)
        ";
        // workaround using sample and hold
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_no_direct_access_possible() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_1hz_meet() {
        let spec = "input i: Int64\noutput a @ 5Hz := 42\noutput b @ 2Hz := 1337\noutput c := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")].expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
    }

    #[test]
    fn test_0_1hz_meet() {
        let spec = "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 0.3Hz := 1337\noutput c := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")].expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_f32(0.1).unwrap()))
        );
    }

    #[test]
    fn test_annotated_freq() {
        let spec = "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 3Hz := 1337\noutput c @2Hz := a + b";
        assert_eq!(num_errors(spec), 1);
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\ninput i: UInt8\noutput o: Float32 @i := sin(2.0)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "o")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_tuple() {
        let spec = "input i: UInt8\noutput out: (Int8, Bool) @i:= (14, false)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "out")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "out")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "in")))
        );
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "b")].expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "a")))
        );
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "out")].expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap()))
        );
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_invalid_op_in_ac() {
        let spec = "input in: Int8\n output out: Int16 @!in := 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_timed() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_timed_faster() {
        let spec = "output o1: Bool @20Hz := false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_timed_incompatible() {
        let spec = "output o1: Bool @3Hz := false\noutput o2: Bool@10Hz:= o1";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_timed_binary() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1 && true";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x.hold().defaults(to: 0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_realtime_offset_not_possible() {
        let spec = "input x: UInt8\n output a := x\noutput b := a.offset(by: 1s)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_realtime_offset_possible() {
        let spec = "input x: UInt8\n output a @1Hz := x.hold()\noutput b := a.offset(by: 1s)";
        assert_eq!(0, num_errors(spec));
    }

    // --------------- Parameterization Tests ---------------------

    #[test]
    fn test_spawn_simple() {
        let spec = "input x:Int8\noutput a (p: Int8) @1Hz spawn with (x) if false := p";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        let exp_a = tt[&NodeId::Expr(hir.expr(get_sr_for_name(&hir, "a")).eid)].clone();

        assert_eq!(exp_a.expression_pacing, ConcretePacingType::Constant);
        assert_eq!(type_a.spawn.0, ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "x"))));
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false)))
        );
    }

    #[test]
    fn test_spawn_simple2() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8, p2:Bool) @1Hz spawn with (x, y) if y := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![])
        );
    }

    #[test]
    fn test_spawn_simple3() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn with (x) if y := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![])
        );
    }

    #[test]
    fn test_spawn_unless() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn with (x) unless y := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::ArithLog(
                ArithLogOp::Not,
                vec![Expression {
                    kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                    eid: ExprId(42),
                    span: Span::Unknown,
                }]
            )
        );
    }

    #[test]
    fn test_spawn_fail() {
        let spec = "input x:Int8\noutput y @1Hz := false\noutput a (p1: Int8) @1Hz spawn with (x) if y := 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated_fail1() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn @x with (x) if y := 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated_fail2() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn @1Hz with (x) if y := 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated() {
        let spec =
            "input x:Int8\ninput y:Bool\ninput z:String\noutput a (p1: Int8) @1Hz spawn @(x&y&z) with (x) if y := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "z"))
            ]))
        );
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![])
        );
    }

    #[test]
    fn test_filter_simple() {
        let spec = "input b:Bool\noutput a filter b := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "b")))
        );
        assert_value_eq!(
            type_a.filter.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![])
        );
    }

    #[test]
    fn test_filter_fail() {
        let spec = "input b:Bool\noutput a @1Hz filter b := 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_fail2() {
        let spec = "input b:Bool\noutput x @1Hz := 5\noutput a filter b := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_simple() {
        let spec = "input b:Bool\noutput a @1Hz close b := 5";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_value_eq!(
            type_a.close.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![])
        );
    }

    #[test]
    fn test_sync_access_wrong_args() {
        let spec = "input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) @x spawn with (x) if y:= 5\n\
            output b(p:Int8) spawn with (x) if y := a(x+5)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_args2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) @x spawn with (x, x) if y:= 5\n\
            output b(p:Int8) spawn with (x) if y := a(x, x+42)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_condition() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) @x spawn with (x, x) if y:= 5\n\
            output b(p:Int8) spawn with (x) if !y := a(x, x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_pacing() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p1:Int8, p2:Int8) @x spawn with (x, x) if y:= 5\n\
            output b(p:Int8) spawn @(z&y) with (z) if y := a(x, x)";
        assert_eq!(1, num_errors(spec));
    }
    #[test]
    fn test_sync_access_missing_spawn() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) @x spawn with (x, x) if y:= 5\n\
            output b := a(x, x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_simple() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) @x spawn with (x) if y:= p\n\
            output b(p:Int8) spawn with (z) if y := a(x)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        let exp_b = tt[&NodeId::Expr(hir.expr(get_sr_for_name(&hir, "b")).eid)].clone();
        assert_value_eq!(
            b.spawn.1,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            }
        );

        assert_eq!(
            b.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "z")),
            ]))
        );

        assert_eq!(
            exp_b.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "x")))
        );

        assert_eq!(
            exp_b.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y")),
            ]))
        );

        assert_value_eq!(
            exp_b.spawn.1,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            }
        );
    }

    #[test]
    fn test_sync_access_filter() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a filter y := x\n\
            output b filter y := a";
        assert_eq!(0, num_errors(spec));
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        assert_eq!(
            b.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            b.filter,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            }
        );
    }

    #[test]
    fn test_sync_access_filter_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a filter y := x\n\
            output b := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_filter_fail2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a filter y := x\n\
            output b filter z := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_close() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a close y := x\n\
            output b close y := a";
        assert_eq!(0, num_errors(spec));
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        assert_value_eq!(
            b.close,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            }
        );
    }

    #[test]
    fn test_sync_access_close_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a close y := x\n\
            output b := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_close_fail2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a close y := x\n\
            output b close z := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_complex() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter !y close z=42 := a(x)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sync_access_complex_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b := a(x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_parameter_no_spawn() {
        let spec = "
            input x:Int8\n\
            output a(p:Int8) := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_target_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with a(x) := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_condition_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with z if a(x) := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b filter a(x) := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b close a(x) := x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_trigger_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            trigger a(x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_parametric_hold_access() {
        let spec = "
            input x:Int8\n\
            output a(p:Int8) spawn with (x) := p + x\n\
            output b := a(x).hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();

        assert_eq!(
            b.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "x")))
        );
        assert_eq!(b.spawn.0, ConcretePacingType::Constant);
        assert_value_eq!(
            b.spawn.1.kind,
            ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true)))
        );
        assert_value_eq!(
            b.filter.kind,
            ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true)))
        );
        assert_value_eq!(
            b.close.kind,
            ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false)))
        );
    }

    #[test]
    fn test_parametric_offset_access() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter !y close z=42 := a(x).offset(by: -1).defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_parametric_addition() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter !y close z=42 := a(x)\n\
            output c spawn @(x&y&z) if y filter !y close z=42 := a(x) + b(z)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_parametric_addition_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter !y close z=1337 := p+42\n\
            output c spawn @(x&y&z) if y filter !y close z=42 := a(x) + b(z)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_missing_spawn() {
        let spec = "
            input i: Bool\n\
            output a @5Hz := 42\n\
            output b @5Hz spawn if i := a
        ";
        assert_eq!(1, num_errors(spec));
    }
}
