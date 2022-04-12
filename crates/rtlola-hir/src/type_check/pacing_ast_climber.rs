use std::collections::HashMap;
use std::rc::Rc;

use rtlola_reporting::{RtLolaError, Span};
use rusttyc::{TcKey, TypeChecker, TypeTable};

use crate::hir::{
    self, AnnotatedPacingType, CloseTemplate, Constant, ExprId, Expression, ExpressionContext, ExpressionKind,
    FnExprKind, Hir, Input, Literal, Offset, Output, SRef, SpawnTemplate, StreamAccessKind, StreamReference, Trigger,
    ValueEq,
};
use crate::modes::HirMode;
use crate::type_check::pacing_types::{
    AbstractPacingType, AbstractSemanticType, ActivationCondition, Freq, InferredTemplates, PacingErrorKind,
    SemanticTypeKind, StreamTypeKeys,
};
use crate::type_check::rtltc::{NodeId, TypeError};
use crate::type_check::{ConcretePacingType, ConcreteStreamPacing};

/// A [Variable] is linked to a reusable [TcKey] in the RustTyc Type Checker.
/// e.g. used to reference stream-variables occurring in multiple places.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Variable(String);

impl rusttyc::TcVar for Variable {}

/// The [PacingTypeChecker] is sued to infer the evaluation rate, as well as close, filter and spawn types for all
/// streams and expressions in the given [Hir].
pub(crate) struct PacingTypeChecker<'a, M>
where
    M: HirMode,
{
    /// The [Hir] to check.
    pub(crate) hir: &'a Hir<M>,
    /// The RustTyc [TypeChecker] used to infer the pacing type.
    pub(crate) pacing_tyc: TypeChecker<AbstractPacingType, Variable>,
    /// A second RustTyc [TypeChecker] instance to infer expression types, e.g. for the close and filter expression.
    pub(crate) expression_tyc: TypeChecker<AbstractSemanticType, Variable>,
    /// Lookup table for stream keys
    pub(crate) node_key: HashMap<NodeId, StreamTypeKeys>,
    /// Maps a RustTyc key of the pacing_tyc to the corresponding span for error reporting.
    pub(crate) pacing_key_span: HashMap<TcKey, Span>,
    /// Maps a RustTyc key of the expression_tyc to the corresponding span for error reporting.
    pub(crate) expression_key_span: HashMap<TcKey, Span>,
    /// Lookup table for the name of a given stream.
    pub(crate) names: &'a HashMap<StreamReference, &'a str>,
    /// Storage to register exact type bounds during Hir climbing, resolved and checked during post process.
    pub(crate) annotated_pacing_checks: HashMap<TcKey, (ConcretePacingType, TcKey)>,
    /// Storage to register exact type bounds during Hir climbing, resolved and checked during post process.
    pub(crate) annotated_exp_checks: HashMap<TcKey, (bool, AbstractSemanticType, TcKey)>,
    /// Expression context providing equivalence for parameters of different streams needed for expression equality.
    pub(crate) exp_context: Rc<ExpressionContext>,
}

impl<'a, M> PacingTypeChecker<'a, M>
where
    M: HirMode + 'static,
{
    /// Creates a new [ValueTypeChecker]. `names`table can be generated from the `Hir`.
    /// Inits all internal hash maps.
    pub(crate) fn new(hir: &'a Hir<M>, names: &'a HashMap<StreamReference, &'a str>) -> Self {
        let node_key = HashMap::new();

        let exp_context = Rc::new(ExpressionContext::new(hir));
        let pacing_tyc = TypeChecker::new();
        let expression_tyc = TypeChecker::new();
        let pacing_key_span = HashMap::new();
        let expression_key_span = HashMap::new();
        let annotated_pacing_checks = HashMap::new();
        let annotated_exp_checks = HashMap::new();
        let mut res = PacingTypeChecker {
            hir,
            pacing_tyc,
            expression_tyc,
            node_key,
            pacing_key_span,
            expression_key_span,
            names,
            annotated_pacing_checks,
            annotated_exp_checks,
            exp_context,
        };
        res.generate_keys_for_streams();
        res
    }

    fn new_stream_key(&mut self) -> StreamTypeKeys {
        let close = self.expression_tyc.new_term_key();
        let spawn = self.expression_tyc.new_term_key();
        let filter = self.expression_tyc.new_term_key();
        self.expression_tyc
            .impose(close.concretizes_explicit(AbstractSemanticType::Negative(SemanticTypeKind::Any)))
            .expect("close key cannot be bound otherwise yet");
        self.expression_tyc
            .impose(spawn.concretizes_explicit(AbstractSemanticType::Positive(SemanticTypeKind::Any)))
            .expect("close key cannot be bound otherwise yet");
        self.expression_tyc
            .impose(filter.concretizes_explicit(AbstractSemanticType::Positive(SemanticTypeKind::Any)))
            .expect("close key cannot be bound otherwise yet");
        StreamTypeKeys {
            exp_pacing: self.pacing_tyc.new_term_key(),
            spawn: (self.pacing_tyc.new_term_key(), spawn),
            filter,
            close,
        }
    }

    fn add_span_to_stream_key(&mut self, keys: StreamTypeKeys, span: Span) {
        self.pacing_key_span.insert(keys.exp_pacing, span.clone());
        self.pacing_key_span.insert(keys.spawn.0, span.clone());
        self.expression_key_span.insert(keys.spawn.1, span.clone());
        self.expression_key_span.insert(keys.filter, span.clone());
        self.expression_key_span.insert(keys.close, span);
    }

    fn impose_more_concrete(
        &mut self,
        keys_l: StreamTypeKeys,
        keys_r: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        self.pacing_tyc
            .impose(keys_l.exp_pacing.concretizes(keys_r.exp_pacing))?;
        self.pacing_tyc.impose(keys_l.spawn.0.concretizes(keys_r.spawn.0))?;
        self.expression_tyc.impose(keys_l.spawn.1.concretizes(keys_r.spawn.1))?;
        self.expression_tyc.impose(keys_l.filter.concretizes(keys_r.filter))?;
        self.expression_tyc.impose(keys_l.close.concretizes(keys_r.close))?;
        Ok(())
    }

    fn generate_keys_for_streams(&mut self) {
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
            self.pacing_key_span
                .insert(key.exp_pacing, self.hir.expr(output.sr).span.clone());
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
                output
                    .instance_template
                    .filter
                    .map(|id| self.hir.expression(id).span.clone())
                    .unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.close,
                output
                    .instance_template
                    .close
                    .as_ref()
                    .map(|ct| self.hir.expression(ct.target).span.clone())
                    .unwrap_or(Span::Unknown),
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

    /// Binds the key to the given annotated pacing type
    fn bind_to_annotated_pacing_type(
        &mut self,
        target: TcKey,
        bound: &AnnotatedPacingType,
        conflict_key: TcKey,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let concrete_pacing = ConcretePacingType::from_pt(bound, self.hir)?;
        self.annotated_pacing_checks
            .insert(target, (concrete_pacing, conflict_key));
        Ok(())
    }

    /// Binds the key to the given annotated expression type
    fn bind_to_annotated_exp_type(&mut self, target: TcKey, bound: &Expression, conflict_key: TcKey, is_close: bool) {
        let parsed = if is_close {
            AbstractSemanticType::for_close(bound, self.exp_context.clone())
        } else {
            AbstractSemanticType::for_filter(bound, self.exp_context.clone())
        };
        self.annotated_exp_checks
            .insert(target, (is_close, parsed, conflict_key));
    }

    fn input_infer(&mut self, input: &Input) -> Result<(), TypeError<PacingErrorKind>> {
        let ac = AbstractPacingType::Event(ActivationCondition::Stream(input.sr));
        let keys = self.node_key[&NodeId::SRef(input.sr)];
        self.pacing_tyc.impose(keys.exp_pacing.concretizes_explicit(ac))?;
        Ok(())
    }

    fn trigger_infer(&mut self, trigger: &Trigger) -> Result<(), TypeError<PacingErrorKind>> {
        let stream_keys = self.node_key[&NodeId::SRef(trigger.sr)];
        let exp_key = self.expression_infer(self.hir.expr(trigger.sr))?;
        // Check if there is a type is annotated
        if let Some(ac) = &trigger.annotated_pacing_type {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.exp_pacing, span);

            self.bind_to_annotated_pacing_type(stream_keys.exp_pacing, ac, exp_key.exp_pacing)?;
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.concretizes_explicit(annotated_ty))?;
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.concretizes(exp_key.exp_pacing))?;
        } else {
            // Trigger type is equal to inferred type
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.equate_with(exp_key.exp_pacing))?;
        }
        Ok(())
    }

    fn output_infer(&mut self, output: &Output) -> Result<(), TypeError<PacingErrorKind>> {
        let stream_keys = self.node_key[&NodeId::SRef(output.sr)];

        // Type Expression Pacing
        let exp_key = self.expression_infer(self.hir.expr(output.sr))?;

        // Check if there is a type is annotated
        if let Some(ac) = &output.annotated_pacing_type {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.exp_pacing, span);

            self.bind_to_annotated_pacing_type(stream_keys.exp_pacing, ac, exp_key.exp_pacing)?;
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.concretizes_explicit(annotated_ty))?;
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.concretizes(exp_key.exp_pacing))?;
        } else {
            // Output type is equal to inferred type
            self.pacing_tyc
                .impose(stream_keys.exp_pacing.equate_with(exp_key.exp_pacing))?;
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
        if let Some(close) = output.instance_template.close.as_ref() {
            self.close_infer(close, stream_keys, exp_key)?;
        }
        Ok(())
    }

    fn spawn_infer(
        &mut self,
        spawn: &SpawnTemplate,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let spawn_target_keys = self.new_stream_key();
        let spawn_condition_keys = self.new_stream_key();

        // Check if there is a pacing annotated
        if let Some(ac) = spawn.pacing.as_ref() {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.spawn.0, span);
            self.pacing_tyc
                .impose(stream_keys.spawn.0.concretizes_explicit(annotated_ty))?;
            self.bind_to_annotated_pacing_type(stream_keys.spawn.0, ac, spawn_target_keys.exp_pacing)?;
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
        let spawn_condition_exp = spawn.condition.map(|eid| self.hir.expression(eid));
        if let Some(spawn_condition) = spawn_condition_exp {
            self.node_key
                .insert(NodeId::Expr(spawn_condition.eid), spawn_condition_keys);
            self.add_span_to_stream_key(spawn_condition_keys, spawn_condition.span.clone());
            let inferred = self.expression_infer(spawn_condition)?;
            self.impose_more_concrete(spawn_condition_keys, inferred)?;

            //Streams spawn condition is equal to annotated condition
            self.bind_to_annotated_exp_type(stream_keys.spawn.1, spawn_condition, exp_keys.spawn.1, false);
            self.expression_tyc.impose(
                stream_keys
                    .spawn
                    .1
                    .concretizes_explicit(AbstractSemanticType::for_filter(
                        spawn_condition,
                        self.exp_context.clone(),
                    )),
            )?;
        }

        // Pacing of spawn target is more concrete than pacing of condition
        self.pacing_tyc.impose(
            spawn_target_keys
                .exp_pacing
                .concretizes(spawn_condition_keys.exp_pacing),
        )?;
        // Spawn condition is more concrete than the spawn condition of the expression
        self.expression_tyc
            .impose(stream_keys.spawn.1.concretizes(exp_keys.spawn.1))?;
        // Spawn target is more concrete than the spawn target of the expression
        self.pacing_tyc
            .impose(stream_keys.spawn.0.concretizes(exp_keys.spawn.0))?;
        // Spawn pacing of the stream is more concrete than the spawn pacing of the target
        self.pacing_tyc
            .impose(stream_keys.spawn.0.concretizes(spawn_target_keys.exp_pacing))?;

        Ok(())
    }

    fn filter_infer(
        &mut self,
        filter_id: ExprId,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let filter = self.hir.expression(filter_id);
        let filter_keys = self.expression_infer(filter)?;

        //Pacing of stream is more concrete than pacing of the filter
        self.pacing_tyc
            .impose(stream_keys.exp_pacing.equate_with(filter_keys.exp_pacing))?;

        //Filter is equal to the expression
        self.bind_to_annotated_exp_type(stream_keys.filter, filter, exp_keys.filter, false);
        self.expression_tyc.impose(
            stream_keys
                .filter
                .concretizes_explicit(AbstractSemanticType::for_filter(filter, self.exp_context.clone())),
        )?;

        //Filter of the stream is more concrete than the filter of the streams expression
        self.expression_tyc
            .impose(stream_keys.filter.concretizes(exp_keys.filter))?;

        // Spawn and Close of the filter expression matches the spawn and close of the stream
        self.pacing_tyc
            .impose(filter_keys.spawn.0.concretizes(stream_keys.spawn.0))?;
        self.expression_tyc
            .impose(filter_keys.spawn.1.concretizes(stream_keys.spawn.1))?;
        self.expression_tyc
            .impose(filter_keys.close.concretizes(stream_keys.close))?;
        Ok(())
    }

    fn close_infer(
        &mut self,
        close: &CloseTemplate,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let close_target = self.hir.expression(close.target);
        let close_keys = self.new_stream_key();

        let inferred = self.expression_infer(close_target)?;
        self.impose_more_concrete(close_keys, inferred)?;
        self.node_key.insert(NodeId::Expr(close_target.eid), close_keys);

        // Check if there is a pacing annotated
        if let Some(ac) = close.pacing.as_ref() {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;

            self.pacing_key_span.insert(close_keys.exp_pacing, span);
            self.pacing_tyc
                .impose(close_keys.exp_pacing.concretizes_explicit(annotated_ty))?;
            self.bind_to_annotated_pacing_type(close_keys.exp_pacing, ac, inferred.exp_pacing)?;
        }

        // Close is equal to the expression
        self.bind_to_annotated_exp_type(exp_keys.close, close_target, stream_keys.close, true);
        self.expression_tyc.impose(
            stream_keys
                .close
                .concretizes_explicit(AbstractSemanticType::for_close(close_target, self.exp_context.clone())),
        )?;

        //Close of the streams expression is more concrete than the close of the stream
        self.expression_tyc
            .impose(exp_keys.close.concretizes(stream_keys.close))?;

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
                    self.pacing_tyc.impose(
                        term_keys
                            .exp_pacing
                            .concretizes_explicit(AbstractPacingType::Periodic(Freq::Any)),
                    )?;
                },
                Offset::PastDiscrete(_) | Offset::FutureDiscrete(_) => {},
            }
        }
        Ok(())
    }

    fn expression_infer(&mut self, exp: &Expression) -> Result<StreamTypeKeys, TypeError<PacingErrorKind>> {
        let term_keys: StreamTypeKeys = self.new_stream_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::LoadConstant(_) | ExpressionKind::ParameterAccess(_, _) => {
                //constants have arbitrary pacing type
            },
            ExpressionKind::StreamAccess(sref, kind, args) => {
                let stream_key = self.node_key[&NodeId::SRef(*sref)];
                match kind {
                    StreamAccessKind::Sync | StreamAccessKind::Offset(_) => {
                        self.handle_offset(kind, term_keys)?;
                        self.impose_more_concrete(term_keys, stream_key)?;

                        //Check that arguments are equal to spawn target if parameterized or the parameters for self
                        let target_spawn_args = self
                            .hir
                            .output(*sref)
                            .map(|o| o.instance_template.spawn_arguments(self.hir))
                            .unwrap_or_else(Vec::new);

                        let target_span = match sref {
                            StreamReference::In(_) => self.hir.input(*sref).unwrap().span.clone(),
                            StreamReference::Out(_) => self.hir.output(*sref).unwrap().span.clone(),
                        };

                        if target_spawn_args.len() != args.len() {
                            return Err(PacingErrorKind::ParameterAmountMismatch {
                                target_span,
                                exp_span: exp.span.clone(),
                                given_num: args.len(),
                                expected_num: target_spawn_args.len(),
                            }
                            .into());
                        }
                        if !args.is_empty() {
                            // Check if args contains a non parameter expression
                            let non_param = args
                                .iter()
                                .find(|e| !matches!(e.kind, ExpressionKind::ParameterAccess(_, _)));
                            if let Some(expr) = non_param {
                                return Err(PacingErrorKind::NonParamInSyncAccess(expr.span.clone()).into());
                            }

                            // Check that every parameter in argument corresponds to one with an equal spawn expression of the target
                            for (target_idx, arg) in args.iter().enumerate() {
                                let (current_stream, current_idx) = match arg.kind {
                                    ExpressionKind::ParameterAccess(c, c_idx) => (c, c_idx),
                                    _ => unreachable!(),
                                };
                                if !self.exp_context.matches(current_stream, current_idx, *sref, target_idx) {
                                    let own_spawn_expr = self
                                        .hir
                                        .output(current_stream)
                                        .map(|c| c.instance_template.spawn_arguments(self.hir)[current_idx].clone())
                                        .expect("Target of sync access must have a spawn expression");
                                    return Err(PacingErrorKind::InvalidSyncAccessParameter {
                                        target_span,
                                        target_spawn_expr: target_spawn_args[target_idx].clone(),
                                        own_spawn_expr,
                                        arg: arg.clone(),
                                    }
                                    .into());
                                }
                            }
                        }
                    },
                    StreamAccessKind::Hold => {},
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => {
                        self.pacing_tyc
                            .impose(term_keys.exp_pacing.concretizes_explicit(Periodic(Freq::Any)))?;
                        // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
                    },
                };

                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            },
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr)?;
                let def_key = self.expression_infer(&*default)?;

                self.impose_more_concrete(term_keys, ex_key)?;
                self.impose_more_concrete(term_keys, def_key)?;
            },
            ExpressionKind::ArithLog(_, args) => {
                match args.len() {
                    2 => {
                        let left_key = self.expression_infer(&args[0])?;
                        let right_key = self.expression_infer(&args[1])?;

                        self.impose_more_concrete(term_keys, left_key)?;
                        self.impose_more_concrete(term_keys, right_key)?;
                    },
                    1 => {
                        let ex_key = self.expression_infer(&args[0])?;
                        self.impose_more_concrete(term_keys, ex_key)?;
                    },
                    _ => unreachable!(),
                }
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let cond_key = self.expression_infer(&*condition)?;
                let cons_key = self.expression_infer(&*consequence)?;
                let alt_key = self.expression_infer(&*alternative)?;

                self.impose_more_concrete(term_keys, cond_key)?;
                self.impose_more_concrete(term_keys, cons_key)?;
                self.impose_more_concrete(term_keys, alt_key)?;
            },
            ExpressionKind::Tuple(elements) => {
                for e in elements {
                    let ele_keys = self.expression_infer(e)?;
                    self.impose_more_concrete(term_keys, ele_keys)?;
                }
            },
            ExpressionKind::Function(FnExprKind { args, .. }) => {
                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            },
            ExpressionKind::TupleAccess(t, _) => {
                let exp_key = self.expression_infer(&*t)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            },
            ExpressionKind::Widen(hir::WidenExprKind { expr: inner, .. }) => {
                let exp_key = self.expression_infer(&*inner)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            },
        };
        self.node_key.insert(NodeId::Expr(exp.eid), term_keys);
        self.add_span_to_stream_key(term_keys, exp.span.clone());
        Ok(term_keys)
    }

    fn check_explicit_bounds(
        pacing_checks: HashMap<TcKey, (ConcretePacingType, TcKey)>,
        exp_checks: HashMap<TcKey, (bool, AbstractSemanticType, TcKey)>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &TypeTable<AbstractSemanticType>,
        ctx: Rc<ExpressionContext>,
    ) -> Vec<TypeError<PacingErrorKind>> {
        let pacing_errs = pacing_checks.into_iter().filter_map(|(key, (bound, conflict_key))| {
            let is = pacing_tt[&key].clone();
            let inferred = pacing_tt[&conflict_key].clone();
            if is != bound {
                Some(TypeError {
                    kind: PacingErrorKind::PacingTypeMismatch(bound, inferred),
                    key1: Some(key),
                    key2: Some(conflict_key),
                })
            } else {
                None
            }
        });
        let exp_errs = exp_checks
            .into_iter()
            .filter_map(|(key, (is_close, bound, conflict_key))| {
                if is_close {
                    let is = AbstractSemanticType::for_close(&exp_tt[&key], ctx.clone());
                    if is != bound {
                        Some(TypeError {
                            kind: PacingErrorKind::SemanticTypeMismatch(bound, is),
                            key1: Some(conflict_key),
                            key2: Some(key),
                        })
                    } else {
                        None
                    }
                } else {
                    let is = AbstractSemanticType::for_filter(&exp_tt[&key], ctx.clone());
                    let inferred = AbstractSemanticType::for_filter(&exp_tt[&conflict_key], ctx.clone());
                    if is != bound {
                        Some(TypeError {
                            kind: PacingErrorKind::SemanticTypeMismatch(bound, inferred),
                            key1: Some(key),
                            key2: Some(conflict_key),
                        })
                    } else {
                        None
                    }
                }
            });
        pacing_errs.chain(exp_errs).collect()
    }

    fn is_parameterized(
        keys: StreamTypeKeys,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &TypeTable<AbstractSemanticType>,
        exp_context: &ExpressionContext,
    ) -> bool {
        let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
        let spawn_cond = exp_tt[&keys.spawn.1].clone();
        let filter = exp_tt[&keys.filter].clone();
        let close = exp_tt[&keys.close].clone();

        let kind_true = ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true)));
        let kind_false = ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false)));

        spawn_pacing != ConcretePacingType::Constant
            || spawn_cond.kind.value_neq(&kind_true, exp_context)
            || filter.kind.value_neq(&kind_true, exp_context)
            || close.kind.value_neq(&kind_false, exp_context)
    }

    fn post_process(
        hir: &Hir<M>,
        nid_key: &HashMap<NodeId, StreamTypeKeys>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &TypeTable<AbstractSemanticType>,
        exp_context: &ExpressionContext,
    ) -> Vec<TypeError<PacingErrorKind>> {
        let mut errors = vec![];

        // Check that every periodic stream has a frequency
        let streams: Vec<(SRef, Span)> = hir
            .outputs()
            .map(|o| (o.sr, o.span.clone()))
            .chain(hir.triggers().map(|t| (t.sr, t.span.clone())))
            .collect();
        for (sref, span) in streams {
            let ct = &pacing_tt[&nid_key[&NodeId::SRef(sref)].exp_pacing];
            match ct {
                ConcretePacingType::Periodic => {
                    errors.push(PacingErrorKind::FreqAnnotationNeeded(span).into());
                },
                ConcretePacingType::Constant => {
                    errors.push(PacingErrorKind::NeverEval(span).into());
                },
                _ => {},
            }
        }

        //Check that trigger expression does not access parameterized stream
        for trigger in hir.triggers() {
            let keys = nid_key[&NodeId::Expr(trigger.expr_id)];
            if Self::is_parameterized(keys, pacing_tt, exp_tt, exp_context) {
                errors.push(
                    PacingErrorKind::ParameterizationNotAllowed(hir.expression(trigger.expr_id).span.clone()).into(),
                );
            }
        }

        //Check that spawn target/condition, filter, close is not again parameterized
        for output in hir.outputs() {
            let output_keys = nid_key[&NodeId::SRef(output.sr)];
            let output_spawn_pacing = pacing_tt[&output_keys.spawn.0].clone();
            let output_spawn_cond = exp_tt[&output_keys.spawn.1].clone();
            let output_filter = exp_tt[&output_keys.filter].clone();
            let output_close = exp_tt[&output_keys.close].clone();

            if let Some(template) = output.instance_template.spawn.as_ref() {
                //Spawn target
                if let Some(target) = template.target {
                    let keys = nid_key[&NodeId::Expr(target)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt, exp_context) {
                        errors.push(
                            PacingErrorKind::ParameterizationNotAllowed(hir.expression(target).span.clone()).into(),
                        );
                    }
                }
                //Spawn condition
                if let Some(condition) = template.condition {
                    let keys = nid_key[&NodeId::Expr(condition)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt, exp_context) {
                        errors.push(
                            PacingErrorKind::ParameterizationNotAllowed(hir.expression(condition).span.clone()).into(),
                        );
                    }
                }
            }
            //Filter expression must have exactly same spawn / close as stream and no filter
            if let Some(filter) = output.instance_template.filter {
                let keys = nid_key[&NodeId::Expr(filter)];

                let kind_true = ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true)));
                if pacing_tt[&keys.spawn.0] != output_spawn_pacing
                    || exp_tt[&keys.spawn.1].value_neq(&output_spawn_cond, exp_context)
                    || exp_tt[&keys.close].value_neq(&output_close, exp_context)
                    || exp_tt[&keys.filter].kind.value_neq(&kind_true, exp_context)
                {
                    errors
                        .push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(filter).span.clone()).into());
                }
            }

            //Close expression must either be non parameterized or have exactly same spawn / filter as stream and no filter
            if let Some(close) = output.instance_template.close.as_ref() {
                let keys = nid_key[&NodeId::Expr(close.target)];
                if Self::is_parameterized(keys, pacing_tt, exp_tt, exp_context)
                    && (pacing_tt[&keys.spawn.0] != output_spawn_pacing
                        || exp_tt[&keys.spawn.1].value_neq(&output_spawn_cond, exp_context)
                        || exp_tt[&keys.filter].value_neq(&output_filter, exp_context)
                        || exp_tt[&keys.close].value_neq(&output_close, exp_context))
                {
                    errors.push(
                        PacingErrorKind::ParameterizationNotAllowed(hir.expression(close.target).span.clone()).into(),
                    );
                }
            }
        }

        //Check that spawn, close pacing is not constant / periodic
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::SRef(output.sr)];
            let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
            if let Some(template) = output.instance_template.spawn.as_ref() {
                if spawn_pacing == ConcretePacingType::Constant || spawn_pacing == ConcretePacingType::Periodic {
                    let span = template
                        .pacing
                        .as_ref()
                        .map(|pt| {
                            match pt {
                                AnnotatedPacingType::Frequency { span, .. } => span.clone(),
                                AnnotatedPacingType::Expr(id) => hir.expression(*id).span.clone(),
                            }
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
            if let Some(close) = output.instance_template.close.as_ref() {
                let keys = nid_key[&NodeId::Expr(close.target)];
                let close_pacing = pacing_tt[&keys.exp_pacing].clone();
                let span = hir.expression(close.target).span.clone();
                if close_pacing == ConcretePacingType::Periodic {
                    errors.push(PacingErrorKind::FreqAnnotationNeeded(span).into())
                } else if close_pacing == ConcretePacingType::Constant {
                    errors.push(PacingErrorKind::NeverEval(span).into())
                }
            }
        }
        //Check that stream without spawn template does not access parameterized stream
        //Check that stream without filter does not access filtered stream
        //Check that stream without close does not access closed stream
        let kind_true = ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true)));
        let kind_false = ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false)));
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::Expr(output.expr_id)];
            let spawn_pacing = pacing_tt[&keys.spawn.0].clone();
            let spawn_cond = exp_tt[&keys.spawn.1].clone();
            let filter_type = exp_tt[&keys.filter].clone();
            let close_type = exp_tt[&keys.close].clone();

            let spawn_pacing = (output.instance_template.spawn.is_none()
                && spawn_pacing != ConcretePacingType::Constant)
                .then(|| spawn_pacing);
            let spawn_cond = (output.instance_template.spawn_condition(hir).is_none()
                && spawn_cond.kind.value_neq(&kind_true, exp_context))
            .then(|| spawn_cond);
            let filter = (output.instance_template.filter.is_none()
                && filter_type.kind.value_neq(&kind_true, exp_context))
            .then(|| filter_type);
            let close = (output.instance_template.close.is_none()
                && close_type.kind.value_neq(&kind_false, exp_context))
            .then(|| close_type);

            if spawn_pacing.is_some() || spawn_cond.is_some() || filter.is_some() || close.is_some() {
                errors.push(
                    PacingErrorKind::ParameterizationNeeded {
                        who: output.span.clone(),
                        why: hir.expression(output.expr_id).span.clone(),
                        inferred: Box::new(InferredTemplates {
                            spawn_pacing,
                            spawn_cond,
                            filter,
                            close,
                        }),
                    }
                    .into(),
                )
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
            if output.annotated_pacing_type.is_none() && exp_pacing != stream_pacing {
                errors.push(PacingErrorKind::UnintuitivePacingWarning(output.span.clone(), stream_pacing).into());
            }
        }

        //Check that no periodic expressions with a spawn access static periodic stream
        let nodes_to_check: Vec<NodeId> = hir
            .outputs
            .iter()
            .flat_map(|o| {
                vec![
                    Some(NodeId::SRef(o.sr)),
                    o.instance_template.filter.map(NodeId::Expr),
                    o.instance_template.close.as_ref().map(|ct| NodeId::Expr(ct.target)),
                ]
            })
            .flatten()
            .collect();

        for node in nodes_to_check {
            let stream_keys = nid_key[&node];
            let exp_pacing = pacing_tt[&stream_keys.exp_pacing].clone();
            let spawn_pacing = pacing_tt[&stream_keys.spawn.0].clone();
            let spawn_cond = exp_tt[&stream_keys.spawn.1].clone();
            if matches!(exp_pacing, ConcretePacingType::FixedPeriodic(_))
                && spawn_pacing != ConcretePacingType::Constant
            {
                let (expr, span) = match node {
                    NodeId::SRef(sr) => {
                        (
                            hir.expr(sr),
                            &hir.output(sr).expect("StreamReference created above is invalid").span,
                        )
                    },
                    NodeId::Expr(eid) => (hir.expression(eid), &hir.expression(eid).span),
                    NodeId::Param(_, _) => unreachable!(),
                };
                let accesses_streams: Vec<StreamReference> = expr.get_sync_accesses();
                for target in accesses_streams {
                    let target_keys = nid_key[&NodeId::SRef(target)];
                    let target_spawn_pacing = pacing_tt[&target_keys.spawn.0].clone();
                    let target_spawn_condition = exp_tt[&target_keys.spawn.1].clone();
                    if spawn_pacing != target_spawn_pacing || spawn_cond.value_neq(&target_spawn_condition, exp_context)
                    {
                        let target_span = hir
                            .outputs()
                            .find(|o| o.sr == target)
                            .map(|o| o.span.clone())
                            .unwrap_or(Span::Unknown);
                        errors.push(
                            PacingErrorKind::SpawnPeriodicMismatch(
                                span.clone(),
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

    /// The callable function to start the inference. Used by [LolaTypeChecker].
    pub(crate) fn type_check(mut self) -> Result<HashMap<NodeId, ConcreteStreamPacing>, RtLolaError> {
        for input in self.hir.inputs() {
            self.input_infer(input)
                .map_err(|e| e.into_diagnostic(&[&self.pacing_key_span, &self.expression_key_span], self.names))?;
        }

        for output in self.hir.outputs() {
            self.output_infer(output)
                .map_err(|e| e.into_diagnostic(&[&self.pacing_key_span, &self.expression_key_span], self.names))?;
        }

        for trigger in self.hir.triggers() {
            self.trigger_infer(trigger)
                .map_err(|e| e.into_diagnostic(&[&self.pacing_key_span, &self.expression_key_span], self.names))?;
        }

        let PacingTypeChecker {
            hir,
            pacing_tyc,
            expression_tyc,
            node_key,
            pacing_key_span,
            expression_key_span,
            names,
            annotated_pacing_checks,
            annotated_exp_checks,
            exp_context,
        } = self;

        let pacing_tt = pacing_tyc.type_check().map_err(|tc_err| {
            TypeError::from(tc_err).into_diagnostic(&[&pacing_key_span, &expression_key_span], names)
        })?;
        let exp_tt = expression_tyc.type_check().map_err(|tc_err| {
            TypeError::from(tc_err).into_diagnostic(&[&pacing_key_span, &expression_key_span], names)
        })?;

        let mut error = RtLolaError::new();
        for pe in Self::check_explicit_bounds(
            annotated_pacing_checks,
            annotated_exp_checks,
            &pacing_tt,
            &exp_tt,
            exp_context.clone(),
        ) {
            error.add(pe.into_diagnostic(&[&pacing_key_span, &expression_key_span], names));
        }
        for pe in Self::post_process(hir, &node_key, &pacing_tt, &exp_tt, exp_context.as_ref()) {
            error.add(pe.into_diagnostic(&[&pacing_key_span, &expression_key_span], names));
        }
        Result::from(error)?;

        let ctt: HashMap<NodeId, ConcreteStreamPacing> = node_key
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

        Ok(ctt)
    }
}

#[cfg(test)]
mod tests {
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use rtlola_parser::ast::RtLolaAst;
    use rtlola_parser::ParserConfig;
    use rtlola_reporting::Span;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    use crate::hir::{
        ArithLogOp, Constant, ExprId, Expression, ExpressionContext, ExpressionKind, Literal, RtLolaHir,
        StreamAccessKind, StreamReference, ValueEq,
    };
    use crate::modes::BaseMode;
    use crate::type_check::pacing_types::ActivationCondition;
    use crate::type_check::rtltc::{LolaTypeChecker, NodeId};
    use crate::type_check::ConcretePacingType;

    macro_rules! assert_value_eq {
        ($left:expr, $right:expr, $ctx: expr) => {{
            match (&$left, &$right) {
                (left_val, right_val) => {
                    if (left_val.value_neq(right_val, &$ctx)) {
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
                },
            }
        }};
    }

    fn setup_ast(spec: &str) -> (RtLolaHir<BaseMode>, ExpressionContext) {
        let ast: RtLolaAst = match rtlola_parser::parse(ParserConfig::for_string(spec.to_string())) {
            Ok(s) => s,
            Err(e) => panic!("Spec {} cannot be parsed: {:?}", spec, e),
        };
        let hir = crate::from_ast(ast).unwrap();
        let ctx = ExpressionContext::new(&hir);
        (hir, ctx)
    }

    fn num_errors(spec: &str) -> usize {
        let (spec, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&spec);
        match ltc.pacing_type_infer() {
            Ok(_) => 0,
            Err(e) => dbg!(e).num_errors(),
        }
    }

    fn get_sr_for_name(hir: &RtLolaHir<BaseMode>, name: &str) -> StreamReference {
        if let Some(i) = hir.get_input_with_name(name) {
            i.sr
        } else {
            hir.get_output_with_name(name).unwrap().sr
        }
    }
    fn get_node_for_name(hir: &RtLolaHir<BaseMode>, name: &str) -> NodeId {
        NodeId::SRef(get_sr_for_name(hir, name))
    }

    #[test]
    fn test_input_simple() {
        let spec = "input i: Int8";
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        let exp_a = tt[&NodeId::Expr(hir.expr(get_sr_for_name(&hir, "a")).eid)].clone();

        assert_eq!(exp_a.expression_pacing, ConcretePacingType::Constant);
        assert_eq!(
            type_a.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "x")))
        );
        assert_value_eq!(
            type_a.spawn.1.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
            ctx
        );
    }

    #[test]
    fn test_spawn_simple2() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8, p2:Bool) @1Hz spawn with (x, y) if y := 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_spawn_simple3() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn with (x) if y := 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_spawn_unless() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) @1Hz spawn with (x) unless y := 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
            ),
            ctx
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_filter_simple() {
        let spec = "input b:Bool\noutput a filter b := 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "b")))
        );
        assert_value_eq!(
            type_a.filter.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![]),
            ctx
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_value_eq!(
            type_a.close.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![]),
            ctx
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
            output a(p:Int8) @x spawn with (x) if y:= p\n\
            output b(p:Int8) spawn with (x) if y := a(p)";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        let exp_b = tt[&NodeId::Expr(hir.expr(get_sr_for_name(&hir, "b")).eid)].clone();
        assert_value_eq!(
            b.spawn.1,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            },
            ctx
        );

        assert_eq!(
            b.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y")),
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
            },
            ctx
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
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
            },
            ctx
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        assert_value_eq!(
            b.close,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            },
            ctx
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
            output b(p:Int8) spawn with (x) if y filter !y close z=42 := a(p)";
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
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();

        assert_eq!(
            b.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "x")))
        );
        assert_eq!(b.spawn.0, ConcretePacingType::Constant);
        assert_value_eq!(
            b.spawn.1.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            b.filter.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            b.close.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
            ctx
        );
    }

    #[test]
    fn test_parametric_offset_access() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (x) if y filter !y close z=42 := a(p).offset(by: -1).defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_parametric_addition() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y filter !y close z=42 := p\n\
            output b(p:Int8) spawn with (x) if y filter !y close z=42 := a(p)\n\
            output c(p) spawn @(x&y&z) with x if y filter !y close z=42 := a(p) + b(p)";
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

    #[test]
    fn test_filter_expr_type() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) if y filter !y close y := p + x
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let filter = tt[&NodeId::Expr(hir.outputs[0].instance_template.filter.unwrap())].clone();
        assert_eq!(
            filter.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_eq!(
            filter.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            filter.spawn.1.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            filter.filter.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            filter.close.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_filter_expr_type2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if y close z=42 := p+x\n\
            output b(p:Int8) spawn with (x) if y filter a(p) close z=42 := p+42";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_expr_type_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (z) if !y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter a(z) close z=1337 := p+42";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_expr_type() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a(p:Int8) spawn with (x) if y filter y close z := p + x
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let close = tt[&NodeId::Expr(hir.outputs[0].instance_template.close.as_ref().unwrap().target)].clone();
        assert_eq!(
            close.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "z")))
        );
        assert_eq!(close.spawn.0, ConcretePacingType::Constant);
        assert_value_eq!(
            close.spawn.1.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            close.filter.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            close.close.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
            ctx
        );
    }

    #[test]
    fn test_close_self() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) if y filter y close a(p) := p + x
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let close = tt[&NodeId::Expr(hir.outputs[0].instance_template.close.as_ref().unwrap().target)].clone();
        assert_eq!(
            close.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_eq!(
            close.spawn.0,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
        assert_value_eq!(
            close.spawn.1.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            close.filter.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            close.close.kind,
            ExpressionKind::StreamAccess(
                get_sr_for_name(&hir, "a"),
                StreamAccessKind::Sync,
                vec![Expression {
                    kind: ExpressionKind::ParameterAccess(get_sr_for_name(&hir, "a"), 0),
                    eid: ExprId(0),
                    span: Span::Unknown
                }]
            ),
            ctx
        );
    }

    #[test]
    fn test_close_expr_type_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) if !y filter y close z=42 := p\n\
            output b(p:Int8) spawn with (z) if y filter !y close a(x) := p+42";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_delay() {
        let spec = "
            input x:Int8\n\
            output a @1Hz spawn if x=42 close if true then true else a := x.aggregate(over: 1s, using: sum) > 1337
        ";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_trigger_annotated() {
        let spec = "
            input x:Bool\n\
            trigger @1Hz x.hold(or: false)
        ";
        assert_eq!(0, num_errors(spec));
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let trigger = tt[&NodeId::SRef(hir.triggers[0].sr)].clone();
        assert_eq!(
            trigger.expression_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
    }

    #[test]
    fn test_trigger_annotated_fail() {
        let spec = "
            input x:Bool\n\
            trigger @1Hz x
        ";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_dynamic_close_static_freq() {
        let spec = "input i: Int8\n\
        output a @1Hz := true\n\
        output b @1Hz spawn if i = 5 close b = 7 & a := 42";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_trigger_annotation_needed() {
        let spec = "
            input x:Bool\n\
            trigger x.hold(or: false)
        ";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_self_ref_counter() {
        let spec = "
            input x: Int32\n\
            output a (p: Int32) @1Hz spawn with x := a(p).offset(by: -1).defaults(to: 0) + 1
        ";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_broken_access() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x if x % 2 == 0 := p + x\n\
                        output b(p:Int8) spawn with y if x % 2 == 0 := a(x) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_broken_access2() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x if x % 2 == 0 := p + x\n\
                        output b(p:Int8) spawn with y if x % 2 == 0 := a(p) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_broken_access3() {
        let spec = "input x:Int8\n\
                        output a(p:Int8) spawn with x if x % 2 == 0 := p + x\n\
                        output b(p:Int8) spawn with x := a(p)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_self_access_faulty() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x := p + x\n\
                        output b(p:Int8) spawn with y := a(p) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_annotated_close_fail() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x close @x y == 5 := p + x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_annotated_close() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x close @x&y y == 5 := p + x";
        assert_eq!(0, num_errors(spec));
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let close = tt[&NodeId::Expr(hir.outputs[0].instance_template.close.as_ref().unwrap().target)].clone();
        assert_eq!(
            close.expression_pacing,
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(get_sr_for_name(&hir, "x")),
                ActivationCondition::Stream(get_sr_for_name(&hir, "y"))
            ]))
        );
    }

    #[test]
    fn test_filter_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x filter a := 5\n\
        output y filter b && c := 42\n\
        output z filter a && b && c := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_mixed() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x filter a && (b || c) := 5\n\
        output y filter a && (b || c) := 42\n\
        output z filter a && (b || c) := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_advanced_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x filter a := 5\n\
        output y filter b && c := 42\n\
        output z := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_advanced_fail2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x filter a := 5\n\
        output y filter b && c := 42\n\
        output z filter d := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x filter a || b:= 5\n\
        output y filter b || c:= 42\n\
        output z filter b := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x filter a || b || d:= 5\n\
        output y filter b || d || c:= 42\n\
        output z filter b || d := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x filter a || b || d:= 5\n\
        output y filter b || d || c:= 42\n\
        output z filter c || d := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a spawn if a := 5\n\
        output y @a spawn if b && c := 42\n\
        output z @a spawn if a && b && c := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a spawn if a := 5\n\
        output y @a spawn if b && c := 42\n\
        output z @a := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced_fail2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x @a spawn if a := 5\n\
        output y @a spawn if b && c := 42\n\
        output z @a spawn if d := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x @a spawn if a || b:= 5\n\
        output y @a spawn if b || c:= 42\n\
        output z @a spawn if b := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x @a spawn if a || b || d:= 5\n\
        output y @a spawn if b || d || c:= 42\n\
        output z @a spawn if b || d := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x @a spawn if a || b || d:= 5\n\
        output y @a spawn if b || d || c:= 42\n\
        output z @a spawn if c || d := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a close a := 5\n\
        output y @a close b || c := 42\n\
        output z @a close a || b || c := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_close_conjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a close a && b := 5\n\
        output y @a close b && c := 42\n\
        output z @a close b := x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_close_conjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a close a && b := 5\n\
        output y @a close b && c := 42\n\
        output z @a close b && c:= x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_advanced_fail2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x @a close a := 5\n\
        output y @a close b || c := 42\n\
        output z @a close a || b := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn close_self_ref() {
        let spec = "input a: Int32\n\
                  output b(p: Bool) spawn with a == 42 filter !p || a == 42 close b(p) == 1337:= a";
        assert_eq!(0, num_errors(spec));
    }
}
