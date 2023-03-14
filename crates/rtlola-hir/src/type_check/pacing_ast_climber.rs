use std::collections::HashMap;
use std::rc::Rc;

use rtlola_reporting::{RtLolaError, Span};
use rusttyc::{Constructable, PreliminaryTypeTable, TcKey, TypeChecker, TypeTable};

use crate::hir::{
    self, AnnotatedPacingType, CloseDef, EvalDef, ExprId, Expression, ExpressionContext, ExpressionKind, FnExprKind,
    Hir, Input, Offset, Output, SRef, SpawnDef, StreamAccessKind, StreamReference, Trigger,
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

/// The [PacingTypeChecker] is used to infer the evaluation rate, as well as close, filter and spawn types for all
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
    /// Storage to register exact type implications during Hir climbing, resolved and checked during post process.
    /// First tuple element should imply the second tuple element.
    pub(crate) annotated_pacing_checks: Vec<(TcKey, TcKey)>,
    /// Storage to register exact type bounds during Hir climbing, resolved and checked during post process.
    /// First tuple element should imply the second tuple element.
    /// Boolean value indicates that the check originates from a close condition.
    pub(crate) annotated_exp_checks: Vec<(TcKey, TcKey, bool)>,
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
        let annotated_pacing_checks = Vec::new();
        let annotated_exp_checks = Vec::new();
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
            eval_pacing: self.pacing_tyc.new_term_key(),
            eval_condition: filter,
            spawn_pacing: self.pacing_tyc.new_term_key(),
            spawn_condition: spawn,
            close_pacing: self.pacing_tyc.new_term_key(),
            close_condition: close,
        }
    }

    fn add_span_to_stream_key(&mut self, keys: StreamTypeKeys, span: Span) {
        self.pacing_key_span.insert(keys.eval_pacing, span);
        self.pacing_key_span.insert(keys.spawn_pacing, span);
        self.pacing_key_span.insert(keys.close_pacing, span);
        self.expression_key_span.insert(keys.spawn_condition, span);
        self.expression_key_span.insert(keys.eval_condition, span);
        self.expression_key_span.insert(keys.close_condition, span);
    }

    fn impose_more_concrete(
        &mut self,
        keys_l: StreamTypeKeys,
        keys_r: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        self.pacing_tyc
            .impose(keys_l.eval_pacing.concretizes(keys_r.eval_pacing))?;
        self.expression_tyc
            .impose(keys_l.eval_condition.concretizes(keys_r.eval_condition))?;
        self.pacing_tyc
            .impose(keys_l.spawn_pacing.concretizes(keys_r.spawn_pacing))?;
        self.expression_tyc
            .impose(keys_l.spawn_condition.concretizes(keys_r.spawn_condition))?;
        self.pacing_tyc
            .impose(keys_l.close_pacing.concretizes(keys_r.close_pacing))?;
        self.expression_tyc
            .impose(keys_l.close_condition.concretizes(keys_r.close_condition))?;
        Ok(())
    }

    fn generate_keys_for_streams(&mut self) {
        for input in self.hir.inputs() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(input.sr), key);
            self.pacing_key_span.insert(key.eval_pacing, input.span);
            self.pacing_key_span.insert(key.spawn_pacing, Span::Unknown);
            self.expression_key_span.insert(key.spawn_condition, Span::Unknown);
            self.expression_key_span.insert(key.eval_condition, Span::Unknown);
            self.expression_key_span.insert(key.close_condition, Span::Unknown);
        }
        for output in self.hir.outputs() {
            let key = self.new_stream_key();
            let spawn_span = self
                .hir
                .spawn(output.sr)
                .map(|spawn| spawn.span)
                .unwrap_or_else(|| Span::Unknown);
            let eval_span = self
                .hir
                .eval(output.sr)
                .map(|eval| eval.span)
                .unwrap_or_else(|| Span::Unknown);
            let close_span = self
                .hir
                .close(output.sr)
                .map(|close| close.span)
                .unwrap_or_else(|| Span::Unknown);

            self.node_key.insert(NodeId::SRef(output.sr), key);
            self.pacing_key_span.insert(key.eval_pacing, eval_span);
            self.pacing_key_span.insert(key.spawn_pacing, spawn_span);
            self.pacing_key_span.insert(key.close_pacing, close_span);
            self.expression_key_span.insert(
                key.spawn_condition,
                output
                    .spawn_cond()
                    .map(|id| self.hir.expression(id).span)
                    .unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.eval_condition,
                output
                    .eval_cond()
                    .map(|id| self.hir.expression(id).span)
                    .unwrap_or(Span::Unknown),
            );
            self.expression_key_span.insert(
                key.close_condition,
                output
                    .close_cond()
                    .map(|cond| self.hir.expression(cond).span)
                    .unwrap_or(Span::Unknown),
            );

            // Create Stream Parameters
            for (idx, parameter) in output.params.iter().enumerate() {
                let key = self.new_stream_key();
                self.node_key.insert(NodeId::Param(idx, output.sr), key);
                self.add_span_to_stream_key(key, parameter.span);
            }
        }

        for trigger in self.hir.triggers() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(trigger.sr), key);
            self.pacing_key_span.insert(key.eval_pacing, trigger.span);
            self.pacing_key_span.insert(key.spawn_pacing, Span::Unknown);
            self.expression_key_span.insert(key.spawn_condition, Span::Unknown);
            self.expression_key_span.insert(key.eval_condition, Span::Unknown);
            self.expression_key_span.insert(key.close_condition, Span::Unknown);
        }
    }

    /// Binds the key to the given annotated pacing type
    fn pacing_type_implies(&mut self, target: TcKey, conflict_key: TcKey) {
        self.annotated_pacing_checks.push((target, conflict_key));
    }

    /// Binds the key to the given annotated expression type
    fn exp_type_implies(&mut self, target: TcKey, conflict_key: TcKey, is_close: bool) {
        self.annotated_exp_checks.push((target, conflict_key, is_close));
    }

    fn input_infer(&mut self, input: &Input) -> Result<(), TypeError<PacingErrorKind>> {
        let ac = AbstractPacingType::Event(ActivationCondition::with_stream(input.sr));
        let keys = self.node_key[&NodeId::SRef(input.sr)];
        self.pacing_tyc.impose(keys.eval_pacing.concretizes_explicit(ac))?;
        Ok(())
    }

    fn trigger_infer(&mut self, trigger: &Trigger) -> Result<(), TypeError<PacingErrorKind>> {
        let stream_keys = self.node_key[&NodeId::SRef(trigger.sr)];
        let exp_key = self.expression_infer(self.hir.eval_unchecked(trigger.sr).expression)?;

        // Check if there is a type is annotated
        if let Some(ac) = &trigger.annotated_pacing_type {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.eval_pacing, span);

            self.pacing_tyc
                .impose(stream_keys.eval_pacing.concretizes_explicit(annotated_ty))?;

            self.pacing_type_implies(stream_keys.eval_pacing, exp_key.eval_pacing);
        } else {
            // Trigger type is equal to inferred type
            self.pacing_tyc
                .impose(stream_keys.eval_pacing.concretizes(exp_key.eval_pacing))?;
        }
        Ok(())
    }

    fn output_infer(&mut self, output: &Output) -> Result<(), TypeError<PacingErrorKind>> {
        // Keys to capture the types of a whole stream
        let stream_keys = self.node_key[&NodeId::SRef(output.sr)];
        let eval = self.hir.eval_unchecked(output.sr);
        let exp_keys = self.expression_infer(eval.expression)?;

        // Type filter
        self.eval_infer(&eval, stream_keys, exp_keys)?;

        // Type spawn condition
        if let Some(spawn) = self.hir.spawn(output.sr) {
            self.spawn_infer(&spawn, stream_keys, exp_keys)?;
        }

        //Type close
        if let Some(close) = self.hir.close(output.sr) {
            self.close_infer(&close, stream_keys, exp_keys)?;
        }
        Ok(())
    }

    fn eval_infer(
        &mut self,
        eval: &EvalDef,
        stream_keys: StreamTypeKeys,
        expr_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        // Keys to capture the types of the eval clause
        let eval_keys = self.new_stream_key();
        self.add_span_to_stream_key(
            eval_keys,
            eval.expression
                .span
                .clone()
                .union(&eval.condition.map(|e| e.span).unwrap_or_else(|| Span::Unknown))
                .union(
                    &eval
                        .annotated_pacing
                        .map(|ap| ap.span(self.hir))
                        .unwrap_or_else(|| Span::Unknown),
                ),
        );

        let filter_keys = eval
            .condition
            .map(|expr| self.expression_infer(expr))
            .unwrap_or_else(|| Ok(self.new_stream_key()))?;

        // eval pacing infer
        self.impose_more_concrete(eval_keys, expr_keys)?;
        self.impose_more_concrete(eval_keys, filter_keys)?;

        if let Some(annotated_pacing) = eval.annotated_pacing {
            let (annotated_ty, span) = AbstractPacingType::from_pt(annotated_pacing, self.hir)?;
            self.pacing_key_span.insert(stream_keys.eval_pacing, span);
            self.pacing_tyc
                .impose(stream_keys.eval_pacing.concretizes_explicit(annotated_ty))?;

            self.pacing_type_implies(stream_keys.eval_pacing, eval_keys.eval_pacing);
        } else {
            self.pacing_tyc
                .impose(stream_keys.eval_pacing.concretizes(eval_keys.eval_pacing))?;
        }

        // eval condition infer
        if let Some(filter) = eval.condition {
            self.expression_tyc.impose(
                stream_keys
                    .eval_condition
                    .concretizes_explicit(AbstractSemanticType::for_filter(filter, self.exp_context.clone())),
            )?;

            // Spawn and Close of the filter expression matches the spawn and close of the stream
            self.pacing_tyc
                .impose(filter_keys.spawn_pacing.concretizes(stream_keys.spawn_pacing))?;
            self.expression_tyc
                .impose(filter_keys.spawn_condition.concretizes(stream_keys.spawn_condition))?;
            self.expression_tyc
                .impose(filter_keys.close_condition.concretizes(stream_keys.close_condition))?;

            // filter implies filter of the expression
            self.exp_type_implies(stream_keys.eval_condition, eval_keys.eval_condition, false);
        } else {
            self.expression_tyc
                .impose(stream_keys.eval_condition.concretizes(eval_keys.eval_condition))?;
        }

        Ok(())
    }

    fn spawn_infer(
        &mut self,
        spawn: &SpawnDef,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        // Keys to capture the types of the eval clause
        let spawn_keys = self.new_stream_key();

        self.add_span_to_stream_key(
            spawn_keys,
            spawn
                .expression
                .map(|e| e.span)
                .unwrap_or_else(|| Span::Unknown)
                .clone()
                .union(&spawn.condition.map(|e| e.span).unwrap_or_else(|| Span::Unknown))
                .union(
                    &spawn
                        .annotated_pacing
                        .map(|ap| ap.span(self.hir))
                        .unwrap_or_else(|| Span::Unknown),
                ),
        );

        let spawn_expr_keys = spawn
            .expression
            .map(|expr| self.expression_infer(expr))
            .unwrap_or_else(|| Ok(self.new_stream_key()))?;
        let spawn_condition_keys = spawn
            .condition
            .map(|expr| self.expression_infer(expr))
            .unwrap_or_else(|| Ok(self.new_stream_key()))?;

        self.impose_more_concrete(spawn_keys, spawn_expr_keys)?;
        self.impose_more_concrete(spawn_keys, spawn_condition_keys)?;

        // spawn pacing
        if let Some(ac) = spawn.annotated_pacing {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.spawn_pacing, span);
            self.pacing_tyc
                .impose(stream_keys.spawn_pacing.concretizes_explicit(annotated_ty))?;
            self.pacing_type_implies(stream_keys.spawn_pacing, spawn_keys.eval_pacing);
        } else {
            self.pacing_tyc
                .impose(stream_keys.spawn_pacing.concretizes(spawn_keys.eval_pacing))?;
        }
        // Spawn expression pacing implies the spawn pacing of the expression
        self.pacing_type_implies(stream_keys.spawn_pacing, exp_keys.spawn_pacing);

        // spawn condition
        if let Some(condition) = spawn.condition {
            self.expression_tyc.impose(
                stream_keys
                    .spawn_condition
                    .concretizes_explicit(AbstractSemanticType::for_filter(condition, self.exp_context.clone())),
            )?;
            // Spawn condition is more concrete than the spawn condition of the expression
            self.exp_type_implies(stream_keys.spawn_condition, exp_keys.spawn_condition, false);
        } else {
            self.expression_tyc
                .impose(stream_keys.spawn_condition.concretizes(exp_keys.spawn_condition))?;
        }

        Ok(())
    }

    fn close_infer(
        &mut self,
        close: &CloseDef,
        stream_keys: StreamTypeKeys,
        exp_keys: StreamTypeKeys,
    ) -> Result<(), TypeError<PacingErrorKind>> {
        let close_keys = close
            .condition
            .map(|expr| self.expression_infer(expr))
            .unwrap_or_else(|| Ok(self.new_stream_key()))?;
        self.add_span_to_stream_key(
            close_keys,
            close
                .condition
                .map(|e| e.span)
                .unwrap_or_else(|| Span::Unknown)
                .clone()
                .union(
                    &close
                        .annotated_pacing
                        .map(|ap| ap.span(self.hir))
                        .unwrap_or_else(|| Span::Unknown),
                ),
        );

        // close pacing
        if let Some(ac) = close.annotated_pacing {
            let (annotated_ty, span) = AbstractPacingType::from_pt(ac, self.hir)?;
            self.pacing_key_span.insert(stream_keys.close_pacing, span);
            self.pacing_tyc
                .impose(stream_keys.close_pacing.concretizes_explicit(annotated_ty))?;
            self.pacing_type_implies(stream_keys.close_pacing, close_keys.eval_pacing);
        } else {
            self.pacing_tyc
                .impose(stream_keys.close_pacing.concretizes(close_keys.eval_pacing))?;
        }

        // close condition
        if let Some(condition) = close.condition {
            //Streams close condition is equal to annotated condition
            self.expression_tyc.impose(
                stream_keys
                    .close_condition
                    .concretizes_explicit(AbstractSemanticType::for_close(condition, self.exp_context.clone())),
            )?;
            //Close of the streams expression is more concrete than the close of the stream
            self.exp_type_implies(stream_keys.close_condition, exp_keys.close_condition, true);
        } else {
            self.expression_tyc
                .impose(stream_keys.close_condition.concretizes(exp_keys.close_condition))?;
        }

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
                            .eval_pacing
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
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::Sync | StreamAccessKind::Offset(_) => {
                        self.handle_offset(kind, term_keys)?;
                        self.impose_more_concrete(term_keys, stream_key)?;

                        //Check that arguments are equal to spawn target if parameterized or the parameters for self
                        let target_spawn_args = self
                            .hir
                            .output(*sref)
                            .and_then(|o| o.spawn())
                            .map(|st| st.spawn_args(self.hir))
                            .unwrap_or_else(Vec::new);

                        let target_span = match sref {
                            StreamReference::In(_) => self.hir.input(*sref).unwrap().span,
                            StreamReference::Out(_) => self.hir.output(*sref).unwrap().span,
                        };

                        if target_spawn_args.len() != args.len() {
                            return Err(PacingErrorKind::ParameterAmountMismatch {
                                target_span,
                                exp_span: exp.span,
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
                                return Err(PacingErrorKind::NonParamInSyncAccess(expr.span).into());
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
                                        .and_then(|o| o.spawn())
                                        .map(|st| st.spawn_args(self.hir)[current_idx].clone())
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
                    StreamAccessKind::Hold | StreamAccessKind::Get | StreamAccessKind::Fresh => {},
                    StreamAccessKind::SlidingWindow(_) => {
                        self.pacing_tyc
                            .impose(term_keys.eval_pacing.concretizes_explicit(Periodic(Freq::Any)))?;
                        // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
                    },
                };

                for arg in args {
                    let arg_key = self.expression_infer(arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            },
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(expr)?;
                let def_key = self.expression_infer(default)?;

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
                let cond_key = self.expression_infer(condition)?;
                let cons_key = self.expression_infer(consequence)?;
                let alt_key = self.expression_infer(alternative)?;

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
                    let arg_key = self.expression_infer(arg)?;
                    self.impose_more_concrete(term_keys, arg_key)?;
                }
            },
            ExpressionKind::TupleAccess(t, _) => {
                let exp_key = self.expression_infer(t)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            },
            ExpressionKind::Widen(hir::WidenExprKind { expr: inner, .. }) => {
                let exp_key = self.expression_infer(inner)?;
                self.impose_more_concrete(term_keys, exp_key)?;
            },
        };
        self.node_key.insert(NodeId::Expr(exp.eid), term_keys);
        self.add_span_to_stream_key(term_keys, exp.span);
        Ok(term_keys)
    }

    fn check_explicit_bounds(
        pacing_checks: Vec<(TcKey, TcKey)>,
        exp_checks: Vec<(TcKey, TcKey, bool)>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &PreliminaryTypeTable<AbstractSemanticType>,
    ) -> Vec<TypeError<PacingErrorKind>> {
        let pacing_errs = pacing_checks.into_iter().filter_map(|(left, right)| {
            let is = pacing_tt[&left].clone();
            let inferred = pacing_tt[&right].clone();
            if !is.implies(&inferred) {
                Some(TypeError {
                    kind: PacingErrorKind::PacingTypeMismatch(is, inferred),
                    key1: Some(left),
                    key2: Some(right),
                })
            } else {
                None
            }
        });
        let exp_errs = exp_checks.into_iter().filter_map(|(left, right, is_close)| {
            let is = &exp_tt[&left].variant;
            let inferred = &exp_tt[&right].variant;
            if (is_close && !inferred.implies(is)) || (!is_close && !is.implies(inferred)) {
                Some(TypeError {
                    kind: PacingErrorKind::SemanticTypeMismatch(is.clone(), inferred.clone()),
                    key1: Some(left),
                    key2: Some(right),
                })
            } else {
                None
            }
        });
        pacing_errs.chain(exp_errs).collect()
    }

    fn is_parameterized(
        keys: StreamTypeKeys,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &PreliminaryTypeTable<AbstractSemanticType>,
    ) -> bool {
        let spawn_pacing = pacing_tt[&keys.spawn_pacing].clone();
        let spawn_cond = &exp_tt[&keys.spawn_condition].variant;
        let filter = &exp_tt[&keys.eval_condition].variant;
        let close = &exp_tt[&keys.close_condition].variant;

        let positive_top = AbstractSemanticType::positive_top();
        let negative_top = AbstractSemanticType::negative_top();
        spawn_pacing != ConcretePacingType::Constant
            || spawn_cond != &positive_top
            || filter != &positive_top
            || close != &negative_top
    }

    fn get_or_fresh_targets(expr: &Expression) -> Vec<(bool, Span, StreamReference)> {
        match &expr.kind {
            ExpressionKind::LoadConstant(_) => vec![],
            ExpressionKind::ArithLog(_, children) => {
                children.iter().flat_map(|e| Self::get_or_fresh_targets(e)).collect()
            },
            ExpressionKind::StreamAccess(target, kind, arguments) => {
                let mut res: Vec<_> = arguments.iter().flat_map(|e| Self::get_or_fresh_targets(e)).collect();
                match kind {
                    StreamAccessKind::Get => res.push((true, expr.span, *target)),
                    StreamAccessKind::Fresh => res.push((false, expr.span, *target)),
                    _ => {},
                };
                res
            },
            ExpressionKind::ParameterAccess(_, _) => vec![],
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let mut cond = Self::get_or_fresh_targets(condition);

                cond.append(&mut Self::get_or_fresh_targets(consequence));
                cond.append(&mut Self::get_or_fresh_targets(alternative));

                cond
            },
            ExpressionKind::Tuple(children) => children.iter().flat_map(|e| Self::get_or_fresh_targets(e)).collect(),
            ExpressionKind::TupleAccess(target, _) => Self::get_or_fresh_targets(target),
            ExpressionKind::Function(def) => def.args.iter().flat_map(|e| Self::get_or_fresh_targets(e)).collect(),
            ExpressionKind::Widen(def) => Self::get_or_fresh_targets(def.expr.as_ref()),
            ExpressionKind::Default { expr, default } => {
                let mut expr = Self::get_or_fresh_targets(expr);

                expr.append(&mut Self::get_or_fresh_targets(default));

                expr
            },
        }
    }

    fn check_get_and_fresh_access(
        hir: &Hir<M>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        nid_key: &HashMap<NodeId, StreamTypeKeys>,
        expr: Option<ExprId>,
        condition: Option<ExprId>,
        own_pacing: &ConcretePacingType,
    ) -> Vec<TypeError<PacingErrorKind>> {
        expr.map(|e| Self::get_or_fresh_targets(hir.expression(e)))
            .unwrap_or_else(Vec::new)
            .iter()
            .chain(
                condition
                    .map(|e| Self::get_or_fresh_targets(hir.expression(e)))
                    .unwrap_or_else(Vec::new)
                    .iter(),
            )
            .flat_map(|(is_get, span, target)| {
                let other_pacing: &ConcretePacingType = &pacing_tt[&nid_key[&NodeId::SRef(*target)].eval_pacing];
                if (own_pacing.is_periodic() != other_pacing.is_periodic())
                    || (own_pacing.is_event_based() != other_pacing.is_event_based())
                {
                    let target_span = hir
                        .output(*target)
                        .map(|o| o.span)
                        .or_else(|| hir.input(*target).map(|i| i.span))
                        .expect("Ref to be either input or output");
                    Some(
                        PacingErrorKind::InvalidGetOrFreshAccess {
                            is_get: *is_get,
                            target: target_span,
                            target_type: other_pacing.clone(),
                            source: *span,
                            source_type: own_pacing.clone(),
                        }
                        .into(),
                    )
                } else {
                    None
                }
            })
            .collect()
    }

    fn post_process(
        hir: &Hir<M>,
        nid_key: &HashMap<NodeId, StreamTypeKeys>,
        pacing_tt: &TypeTable<AbstractPacingType>,
        exp_tt: &PreliminaryTypeTable<AbstractSemanticType>,
    ) -> Vec<TypeError<PacingErrorKind>> {
        let mut errors = vec![];

        // Check that every periodic stream has a frequency
        let streams: Vec<(SRef, Span)> = hir
            .outputs()
            .map(|o| (o.sr, o.span))
            .chain(hir.triggers().map(|t| (t.sr, t.span)))
            .collect();
        for (sref, span) in streams {
            let ct = &pacing_tt[&nid_key[&NodeId::SRef(sref)].eval_pacing];
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
            if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(trigger.expr_id).span).into());
            }
        }

        //Check that spawn target/condition, filter, close is not again parameterized
        for output in hir.outputs() {
            let output_keys = nid_key[&NodeId::SRef(output.sr)];
            let output_spawn_pacing = pacing_tt[&output_keys.spawn_pacing].clone();
            let output_spawn_cond = &exp_tt[&output_keys.spawn_condition].variant;
            let output_filter = &exp_tt[&output_keys.eval_condition].variant;
            let output_close = &exp_tt[&output_keys.close_condition].variant;

            if let Some(spawn) = output.spawn() {
                //Spawn target
                if let Some(target) = spawn.expression {
                    let keys = nid_key[&NodeId::Expr(target)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                        errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(target).span).into());
                    }
                }
                //Spawn condition
                if let Some(condition) = spawn.condition {
                    let keys = nid_key[&NodeId::Expr(condition)];
                    if Self::is_parameterized(keys, pacing_tt, exp_tt) {
                        errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(condition).span).into());
                    }
                }
            }
            //Filter expression must have exactly same spawn / close as stream and no filter
            if let Some(filter) = output.eval_cond() {
                let keys = nid_key[&NodeId::Expr(filter)];

                let positive_top = AbstractSemanticType::positive_top();
                if pacing_tt[&keys.spawn_pacing] != output_spawn_pacing
                    || &exp_tt[&keys.spawn_condition].variant != output_spawn_cond
                    || &exp_tt[&keys.close_condition].variant != output_close
                    || exp_tt[&keys.eval_condition].variant != positive_top
                {
                    errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(filter).span).into());
                }
            }

            //Close expression must either be non parameterized or have exactly same spawn / filter as stream and no filter
            if let Some(cond) = output.close_cond() {
                let keys = nid_key[&NodeId::Expr(cond)];
                if Self::is_parameterized(keys, pacing_tt, exp_tt)
                    && (pacing_tt[&keys.spawn_pacing] != output_spawn_pacing
                        || &exp_tt[&keys.spawn_condition].variant != output_spawn_cond
                        || &exp_tt[&keys.eval_condition].variant != output_filter
                        || &exp_tt[&keys.close_condition].variant != output_close)
                {
                    errors.push(PacingErrorKind::ParameterizationNotAllowed(hir.expression(cond).span).into());
                }
            }
        }

        //Check that spawn, close pacing is not constant / periodic
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::SRef(output.sr)];
            let spawn_pacing = pacing_tt[&keys.spawn_pacing].clone();
            if let Some(spawn) = output.spawn() {
                if spawn_pacing == ConcretePacingType::Constant || spawn_pacing == ConcretePacingType::Periodic {
                    let span = spawn
                        .pacing
                        .as_ref()
                        .map(|pt| {
                            match pt {
                                AnnotatedPacingType::Frequency { span, .. } => *span,
                                AnnotatedPacingType::Expr(id) => hir.expression(*id).span,
                            }
                        })
                        .or_else(|| spawn.expression.map(|id| hir.expression(id).span))
                        .or_else(|| spawn.condition.map(|id| hir.expression(id).span))
                        .unwrap_or(output.span);
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
            if let Some(cond) = output.close_cond() {
                let close_pacing = pacing_tt[&keys.close_pacing].clone();
                let span = hir.expression(cond).span;
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
        let positive_top = AbstractSemanticType::positive_top();
        let negative_top = AbstractSemanticType::negative_top();
        for output in hir.outputs() {
            let keys = nid_key[&NodeId::Expr(output.eval_expr())];
            let spawn_pacing = pacing_tt[&keys.spawn_pacing].clone();
            let spawn_cond = &exp_tt[&keys.spawn_condition].variant;
            let filter_type = &exp_tt[&keys.eval_condition].variant;
            let close_type = &exp_tt[&keys.close_condition].variant;

            let spawn_pacing =
                (output.spawn().is_none() && spawn_pacing != ConcretePacingType::Constant).then_some(spawn_pacing);
            let spawn_cond = (output.spawn_cond().is_none() && spawn_cond != &positive_top)
                .then(|| spawn_cond.construct(&[]).expect("variant to not be any"));
            let filter = (output.eval_cond().is_none() && filter_type != &positive_top)
                .then(|| filter_type.construct(&[]).expect("variant to not be any"));
            let close = (output.close().is_none() && close_type != &negative_top)
                .then(|| close_type.construct(&[]).expect("variant to not be any"));

            if spawn_pacing.is_some() || spawn_cond.is_some() || filter.is_some() || close.is_some() {
                errors.push(
                    PacingErrorKind::ParameterizationNeeded {
                        who: output.span,
                        why: hir.expression(output.eval_expr()).span,
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
            if let Some(spawn) = output.spawn() {
                if let Some(target_id) = spawn.expression {
                    let target_type = pacing_tt[&nid_key[&NodeId::Expr(target_id)].eval_pacing].clone();
                    let spawn_pacing = pacing_tt[&nid_key[&NodeId::SRef(output.sr)].spawn_pacing].clone();
                    if spawn.pacing.is_none() && target_type != spawn_pacing {
                        errors.push(
                            PacingErrorKind::UnintuitivePacingWarning(hir.expression(target_id).span, spawn_pacing)
                                .into(),
                        );
                    }
                }
            }
        }

        //Warning unintuitive exp pacing
        for output in hir.outputs() {
            let exp_pacing = pacing_tt[&nid_key[&NodeId::Expr(output.eval_expr())].eval_pacing].clone();
            let stream_pacing = pacing_tt[&nid_key[&NodeId::SRef(output.sr)].eval_pacing].clone();
            if output.eval_pacing().is_none() && exp_pacing != stream_pacing {
                errors.push(PacingErrorKind::UnintuitivePacingWarning(output.span, stream_pacing).into());
            }
        }

        //Check that no periodic expressions with a spawn access static periodic stream
        let nodes_to_check: Vec<NodeId> = hir
            .outputs
            .iter()
            .flat_map(|o| {
                vec![
                    Some(NodeId::SRef(o.sr)),
                    o.eval_cond().map(NodeId::Expr),
                    o.close_cond().map(NodeId::Expr),
                ]
            })
            .flatten()
            .collect();

        for node in nodes_to_check {
            let stream_keys = nid_key[&node];
            let exp_pacing = pacing_tt[&stream_keys.eval_pacing].clone();
            let spawn_pacing = pacing_tt[&stream_keys.spawn_pacing].clone();
            let spawn_cond = &exp_tt[&stream_keys.spawn_condition].variant;
            if matches!(exp_pacing, ConcretePacingType::FixedPeriodic(_))
                && spawn_pacing != ConcretePacingType::Constant
            {
                let (expr, span) = match node {
                    NodeId::SRef(sr) => {
                        (
                            hir.eval_unchecked(sr).expression,
                            &hir.output(sr).expect("StreamReference created above is invalid").span,
                        )
                    },
                    NodeId::Expr(eid) => (hir.expression(eid), &hir.expression(eid).span),
                    NodeId::Param(_, _) => unreachable!(),
                };
                let accesses_streams: Vec<StreamReference> = expr.get_sync_accesses();
                for target in accesses_streams {
                    let target_keys = nid_key[&NodeId::SRef(target)];
                    let target_spawn_pacing = pacing_tt[&target_keys.spawn_pacing].clone();
                    let target_spawn_condition = &exp_tt[&target_keys.spawn_condition].variant;
                    if spawn_pacing != target_spawn_pacing || spawn_cond != target_spawn_condition {
                        let target_span = hir
                            .outputs()
                            .find(|o| o.sr == target)
                            .map(|o| o.span)
                            .unwrap_or(Span::Unknown);
                        errors.push(
                            PacingErrorKind::SpawnPeriodicMismatch(
                                *span,
                                target_span,
                                (
                                    spawn_pacing.clone(),
                                    spawn_cond.construct(&[]).expect("Variant not to be Any"),
                                ),
                            )
                            .into(),
                        );
                    }
                }
            }
        }

        // Check that a get / fresh access only occurs between periodic or event-based streams
        for output in &hir.outputs {
            let keys = &nid_key[&NodeId::SRef(output.sr)];
            // Spawn clause
            if let Some(spawn) = &output.spawn {
                let own_pacing: &ConcretePacingType = &pacing_tt[&keys.spawn_pacing];
                errors.append(&mut Self::check_get_and_fresh_access(
                    hir,
                    pacing_tt,
                    nid_key,
                    spawn.expression,
                    spawn.condition,
                    own_pacing,
                ))
            }

            // Eval clause
            let eval = &output.eval;
            let own_pacing: &ConcretePacingType = &pacing_tt[&keys.eval_pacing];
            errors.append(&mut Self::check_get_and_fresh_access(
                hir,
                pacing_tt,
                nid_key,
                Some(eval.expr),
                eval.condition,
                own_pacing,
            ));

            // Close clause
            if let Some(close) = &output.close {
                let own_pacing: &ConcretePacingType = &pacing_tt[&keys.close_pacing];
                errors.append(&mut Self::check_get_and_fresh_access(
                    hir,
                    pacing_tt,
                    nid_key,
                    None,
                    Some(close.condition),
                    own_pacing,
                ));
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
            exp_context: _,
        } = self;

        let pacing_tt = pacing_tyc.type_check().map_err(|tc_err| {
            TypeError::from(tc_err).into_diagnostic(&[&pacing_key_span, &expression_key_span], names)
        })?;
        let preliminary_exp_tt = expression_tyc.clone().type_check_preliminary().map_err(|tc_err| {
            TypeError::from(tc_err).into_diagnostic(&[&pacing_key_span, &expression_key_span], names)
        })?;

        let mut error = RtLolaError::new();
        for pe in Self::check_explicit_bounds(
            annotated_pacing_checks,
            annotated_exp_checks,
            &pacing_tt,
            &preliminary_exp_tt,
        ) {
            error.add(pe.into_diagnostic(&[&pacing_key_span, &expression_key_span], names));
        }
        for pe in Self::post_process(hir, &node_key, &pacing_tt, &preliminary_exp_tt) {
            error.add(pe.into_diagnostic(&[&pacing_key_span, &expression_key_span], names));
        }
        Result::from(error)?;

        let exp_tt = expression_tyc.type_check().map_err(|tc_err| {
            TypeError::from(tc_err).into_diagnostic(&[&pacing_key_span, &expression_key_span], names)
        })?;

        let ctt: HashMap<NodeId, ConcreteStreamPacing> = node_key
            .iter()
            .map(|(id, key)| {
                let eval_pacing = pacing_tt[&key.eval_pacing].clone();
                let spawn_pacing = pacing_tt[&key.spawn_pacing].clone();
                let spawn_condition_expression = exp_tt[&key.spawn_condition].clone();
                let filter = exp_tt[&key.eval_condition].clone();
                let close_pacing = pacing_tt[&key.close_pacing].clone();
                let close = exp_tt[&key.close_condition].clone();

                (
                    *id,
                    ConcreteStreamPacing {
                        eval_pacing,
                        eval_condition: filter,
                        spawn_pacing,
                        spawn_condition: spawn_condition_expression,
                        close_pacing,
                        close_condition: close,
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
        // use rtlola_reporting::Handler;
        // let handler = Handler::from(ParserConfig::for_string(spec.to_string()));
        let (spec, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&spec);
        match ltc.pacing_type_infer() {
            Ok(_) => 0,
            Err(e) => {
                // handler.emit_error(&e);
                e.num_errors()
            },
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
            tt[&get_node_for_name(&hir, "i")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_output_simple() {
        let spec = "input a: Int8\n input b: Int8 \n output o := a + b";
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = get_sr_for_name(&hir, "a");
        let ac_b = get_sr_for_name(&hir, "b");
        assert_eq!(
            tt[&get_node_for_name(&hir, "o")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[ac_a, ac_b]))
        );
    }

    #[test]
    fn test_trigger_simple() {
        let spec = "input a: Int8\n trigger a == 42";
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::with_stream(get_sr_for_name(&hir, "a"));
        assert_eq!(
            tt[&NodeId::SRef(hir.triggers().nth(0).unwrap().sr)].eval_pacing,
            ConcretePacingType::Event(ac_a)
        );
    }

    #[test]
    fn test_disjunction_annotated() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let a = get_sr_for_name(&hir, "a");
        let b = get_sr_for_name(&hir, "b");
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "x")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::Models(
                vec![
                    vec![a].into_iter().collect(),
                    vec![b].into_iter().collect(),
                    vec![a, b].into_iter().collect()
                ]
                .into_iter()
                .collect()
            ))
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
            tt[&get_node_for_name(&hir, "a")].eval_pacing,
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
            tt[&get_node_for_name(&hir, "x")].eval_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap()))
        );
    }

    #[test]
    #[ignore = "optional stream access (.get) not implemented correctly in Hir"]
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
            tt[&x_id].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[a_id, b_id]))
        );

        assert_eq!(
            tt[&y_id].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[a_id, b_id, c_id]))
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
            tt[&get_node_for_name(&hir, "c")].eval_pacing,
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
            tt[&get_node_for_name(&hir, "c")].eval_pacing,
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
            tt[&get_node_for_name(&hir, "o")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "i")))
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
            tt[&get_node_for_name(&hir, "out")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "i")))
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
            tt[&get_node_for_name(&hir, "out")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "in")))
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
            tt[&get_node_for_name(&hir, "b")].eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "a")))
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
            tt[&get_node_for_name(&hir, "out")].eval_pacing,
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
        let spec = "input x:Int8\noutput a (p: Int8) spawn with (x) when false eval @1Hz with p";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        let exp_a = tt[&NodeId::Expr(hir.eval_unchecked(get_sr_for_name(&hir, "a")).expression.eid)].clone();

        assert_eq!(exp_a.eval_pacing, ConcretePacingType::Constant);
        assert_eq!(
            type_a.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "x")))
        );
        assert_value_eq!(
            type_a.spawn_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
            ctx
        );
    }

    #[test]
    fn test_spawn_simple2() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8, p2:Bool) spawn with (x, y) when y eval @1Hz with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            type_a.spawn_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_spawn_simple3() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) spawn with (x) when y eval @1Hz with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            type_a.spawn_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_spawn_unless() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) spawn with (x) when !y eval @1Hz with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            type_a.spawn_condition.kind,
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
        let spec = "input x:Int8\noutput y @1Hz := false\noutput a (p1: Int8) spawn with (x) when y eval @1Hz with 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated_fail1() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) spawn @x with (x) when y eval @1Hz with 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated_fail2() {
        let spec = "input x:Int8\ninput y:Bool\noutput a (p1: Int8) spawn @1Hz with (x) when y eval @1Hz with 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_annotated() {
        let spec =
            "input x:Int8\ninput y:Bool\ninput z:String\noutput a (p1: Int8) spawn @(x&y&z) with (x) when y eval @1Hz with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y"),
                get_sr_for_name(&hir, "z")
            ]))
        );
        assert_value_eq!(
            type_a.spawn_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_filter_simple() {
        let spec = "input b:Bool\noutput a eval when b with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_eq!(
            type_a.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "b")))
        );
        assert_value_eq!(
            type_a.eval_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_filter_fail() {
        let spec = "input b:Bool\noutput a eval @1Hz when b with 5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_fail2() {
        let spec = "input b:Bool\noutput x @1Hz := 5\noutput a eval when b with x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_simple() {
        let spec = "input b:Bool\noutput a close when b eval @1Hz with 5";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));

        let type_a = tt[&get_node_for_name(&hir, "a")].clone();
        assert_value_eq!(
            type_a.close_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "b"), StreamAccessKind::Sync, vec![]),
            ctx
        );
    }

    #[test]
    fn test_sync_access_wrong_args() {
        let spec = "input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) when y eval @x with 5\n\
            output b(p:Int8) spawn with (x) when y eval with a(x+5)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_args2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) spawn with (x, x) when y eval @x with 5\n\
            output b(p:Int8) spawn with (x) when y eval with a(x, x+42)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_condition() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) spawn with (x, x) when y eval @x with 5\n\
            output b(p:Int8) spawn with (x) when !y eval with a(x, x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_wrong_pacing() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p1:Int8, p2:Int8) spawn with (x, x) when y eval @x with 5\n\
            output b(p:Int8) spawn @(z&y) with (z) when y eval with a(x, x)";
        assert_eq!(1, num_errors(spec));
    }
    #[test]
    fn test_sync_access_missing_spawn() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p1:Int8, p2:Int8) spawn with (x, x) when y eval @x with 5\n\
            output b := a(x, x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_simple() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) when y eval @x with p\n\
            output b(p:Int8) spawn with (x) when y eval with a(p)";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        let exp_b = tt[&NodeId::Expr(hir.eval_unchecked(get_sr_for_name(&hir, "b")).expression.eid)].clone();
        assert_value_eq!(
            b.spawn_condition,
            Expression {
                kind: ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
                eid: ExprId(42),
                span: Span::Unknown
            },
            ctx
        );

        assert_eq!(
            b.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y"),
            ]))
        );

        assert_eq!(
            exp_b.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "x")))
        );

        assert_eq!(
            exp_b.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y"),
            ]))
        );

        assert_value_eq!(
            exp_b.spawn_condition,
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
            output a eval when y with x\n\
            output b eval when y with a";
        assert_eq!(0, num_errors(spec));
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        assert_eq!(
            b.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            b.eval_condition,
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
            output a eval when y with x\n\
            output b := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_filter_fail2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a eval when y with x\n\
            output b eval when z with a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_close() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a close when y eval with x\n\
            output b close when y eval with a";
        assert_eq!(0, num_errors(spec));
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();
        assert_value_eq!(
            b.close_condition,
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
            output a close when y eval with x\n\
            output b := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_close_fail2() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a close when y eval with x\n\
            output b close when z eval with a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sync_access_complex() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y close when z=42 eval when !y with p\n\
            output b(p:Int8) spawn with (x) when y close when z=42 eval when !y with a(p)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sync_access_complex_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y close when z=42 eval when !y with p\n\
            output b := a(x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_target_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y close when z=42 eval when !y with p\n\
            output b(p:Int8) spawn with a(x) eval with x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_condition_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y close when z=42 eval when !y with p\n\
            output b(p:Int8) spawn with z when a(x) eval with x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y eval when !y with p close when z=42\n\
            output b eval when a(x) with x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y eval when !y with p close when z=42\n\
            output b close when a(x) eval with x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_trigger_parameterized() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y eval when !y with p close when z=42\n\
            trigger a(x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_parametric_hold_access() {
        let spec = "
            input x:Int8\n\
            output a(p:Int8) spawn with (x) eval with p + x\n\
            output b := a(x).hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&get_node_for_name(&hir, "b")].clone();

        assert_eq!(
            b.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "x")))
        );
        assert_eq!(b.spawn_pacing, ConcretePacingType::Constant);
        assert_value_eq!(
            b.spawn_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            b.eval_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            b.close_condition.kind,
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
            output a(p:Int8) spawn with (x) when y eval when !y with p close when z=42\n\
            output b(p:Int8) spawn with (x) when y eval when !y with a(p).offset(by: -1).defaults(to: 0) close when z=42";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_parametric_addition() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y close when z=42 eval when !y with p\n\
            output b(p:Int8) spawn with (x) when y close when z=42 eval when !y with a(p)\n\
            output c(p) spawn @(x&y&z) with x when y close when z=42 eval when !y with a(p) + b(p)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_parametric_addition_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (x) when y eval when !y with p close when z=42\n\
            output b(p:Int8) spawn with (z) when y eval when !y with p+42 close when z=1337 \n\
            output c spawn @(x&y&z) when y close when z=42  eval when !y with a(x) + b(z)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_missing_spawn() {
        let spec = "
            input i: Bool\n\
            output a @5Hz := 42\n\
            output b spawn when i eval @5Hz with a
        ";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_expr_type() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) when y close when y eval when !y with p + x
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let filter = tt[&NodeId::Expr(hir.outputs[0].eval_cond().unwrap())].clone();
        assert_eq!(
            filter.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "y")))
        );
        assert_eq!(
            filter.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            filter.spawn_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            filter.eval_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            filter.close_condition.kind,
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
            output a(p:Int8) spawn with (x) when y close when z=42 eval with p+x\n\
            output b(p:Int8) spawn with (x) when y close when z=42 eval when a(p) with p+42";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_expr_type_fail() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Int8\n\
            output a(p:Int8) spawn with (z) when !y close when z=42 eval with p\n\
            output b(p:Int8) spawn with (z) when y close when z=1337 eval when a(z) with p+42";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_expr_type() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            input z:Bool\n\
            output a(p:Int8) spawn with (x) when y eval when y with p + x close when z
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let close = tt[&NodeId::Expr(hir.outputs[0].close_cond().unwrap())].clone();
        assert_eq!(
            close.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(get_sr_for_name(&hir, "z")))
        );
        assert_eq!(close.spawn_pacing, ConcretePacingType::Constant);
        assert_value_eq!(
            close.spawn_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            close.eval_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
            ctx
        );
        assert_value_eq!(
            close.close_condition.kind,
            ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
            ctx
        );
    }

    #[test]
    fn test_close_self() {
        let spec = "
            input x:Int8\n\
            input y:Bool\n\
            output a(p:Int8) spawn with (x) when y eval when y with p + x close when a(p)
        ";
        let (hir, ctx) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let close = tt[&NodeId::Expr(hir.outputs[0].close_cond().unwrap())].clone();
        assert_eq!(
            close.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_eq!(
            close.spawn_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
        assert_value_eq!(
            close.spawn_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            close.eval_condition.kind,
            ExpressionKind::StreamAccess(get_sr_for_name(&hir, "y"), StreamAccessKind::Sync, vec![]),
            ctx
        );
        assert_value_eq!(
            close.close_condition.kind,
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
            output a(p:Int8) spawn with (x) when !y eval when y with p close when z=42\n\
            output b(p:Int8) spawn with (z) when y eval when !y with p+42 close when a(x)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_delay() {
        let spec = "
            input x:Int8\n\
            output a spawn when x=42 close when if true then true else a eval @1Hz with x.aggregate(over: 1s, using: sum) > 1337
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
            trigger.eval_pacing,
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
        output b spawn when i = 5 close when b = 7 & a eval @1Hz with 42";
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
            output a (p: Int32) spawn with x eval @1Hz with a(p).offset(by: -1).defaults(to: 0) + 1
        ";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_broken_access() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x when x % 2 == 0 eval with p + x\n\
                        output b(p:Int8) spawn with y when x % 2 == 0 eval with a(x) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_broken_access2() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x when x % 2 == 0 eval with p + x\n\
                        output b(p:Int8) spawn with y when x % 2 == 0 eval with a(p) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_broken_access3() {
        let spec = "input x:Int8\n\
                        output a(p:Int8) spawn with x when x % 2 == 0 eval with p + x\n\
                        output b(p:Int8) spawn with x eval with a(p)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_self_access_faulty() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x eval with p + x\n\
                        output b(p:Int8) spawn with y eval with a(p) + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_annotated_close_fail() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x close @x when y == 5 eval with p + x";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_annotated_close() {
        let spec = "input x:Int8\n\
                        input y:Int8\n\
                        output a(p:Int8) spawn with x close @x&y when y == 5 eval with p + x";
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let tys = tt[&NodeId::SRef(hir.outputs[0].sr)].clone();
        assert_eq!(
            tys.close_pacing,
            ConcretePacingType::Event(ActivationCondition::conjunction(&[
                get_sr_for_name(&hir, "x"),
                get_sr_for_name(&hir, "y")
            ]))
        );
    }

    #[test]
    fn test_filter_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x eval when a with 5\n\
        output y eval when b && c with 42\n\
        output z eval when a && b && c with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_mixed() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x eval when a && (b || c) with 5\n\
        output y eval when a && (b || c) with 42\n\
        output z eval when a && (b || c) with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_advanced_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x eval when a with 5\n\
        output y eval when b && c with 42\n\
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
        output x eval when a with 5\n\
        output y eval when b && c with 42\n\
        output z eval when d with x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x eval when a || b with 5\n\
        output y eval when b || c with 42\n\
        output z eval when b with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x eval when a || b || d with 5\n\
        output y eval when b || d || c with 42\n\
        output z eval when b || d with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_filter_disjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x eval when a || b || d with 5\n\
        output y eval when b || d || c with 42\n\
        output z eval when c || d with x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x spawn when a eval @a with 5\n\
        output y spawn when b && c eval @a with 42\n\
        output z spawn when a && b && c eval @a with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x spawn when a eval @a with 5\n\
        output y spawn when b && c eval @a with 42\n\
        output z := x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_advanced_fail2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x spawn when a eval @a with 5\n\
        output y spawn when b && c eval @a with 42\n\
        output z spawn when d eval @a with x + y";
        assert_eq!(2, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x spawn when a || b eval @a with 5\n\
        output y spawn when b || c eval @a with 42\n\
        output z spawn @(a & b & c) when b eval @a with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x spawn when a || b || d eval @a with 5\n\
        output y spawn when b || d || c eval @a with 42\n\
        output z spawn @(a ∧ b ∧ c ∧ d) when b || d eval @a with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_spawn_cond_disjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        input d:Bool\n\
        output x spawn when a || b || d eval @a with 5\n\
        output y spawn when b || d || c eval @a with 42\n\
        output z spawn @(a&b&c&d) when c || d eval @a with x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_advanced() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x close when a eval @a with 5\n\
        output y close when b || c eval @a with 42\n\
        output z close when a || b || c eval @a with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_close_conjunction() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x close when a && b eval @a with 5\n\
        output y close when b && c eval @a with 42\n\
        output z close when b eval @a with x + y";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_close_conjunction_fail() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x close when a && b eval @a with 5\n\
        output y close when b && c eval @a with 42\n\
        output z close when b && c eval @a with x + y";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_advanced_fail2() {
        let spec = "\
        input a:Bool\n\
        input b:Bool\n\
        input c:Bool\n\
        output x close when a eval @a with 5\n\
        output y close when b || c eval @a with 42\n\
        output z close when a || b eval @a with x + y";

        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn close_self_ref() {
        let spec = "input a: Int32\n\
                  output b(p: Bool) spawn with a == 42 close when b(p) == 1337 eval when !p || a == 42 with a";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_missing_eval_with_filter() {
        let spec = "input a: Int8\noutput b eval when a > 3";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_missing_eval_with_filter_faulty() {
        let spec = "input a: Int8\noutput b eval when true";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_missing_eval_with_apt() {
        let spec = "input a: Int8\noutput b eval @1Hz";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_invalid_get() {
        let spec = "input a: Int8\n\
        output b \n\
            eval @1Hz when a.get().defaults(to:42) == 42 with a.get(or: 0)\n\
        output c(p) \n\
            spawn @1Hz with a.get(or: 42) when a.get().defaults(to: 2) == 2\n\
            eval @1Hz with 5\n\
        output d \n\
            eval @1Hz with 5\n\
            close @0.5Hz when a.get(or: 5) == 42";
        assert_eq!(5, num_errors(spec));
    }

    #[test]
    fn test_valid_get() {
        let spec = "input a: Int8\n\
        output b \n\
            eval @a when a.get().defaults(to:42) == 42 with a.get(or: 0)\n\
        output c(p) \n\
            spawn @a with a.get(or: 42) when a.get().defaults(to: 2) == 2\n\
            eval @1Hz with 5\n\
        output d \n\
            eval @1Hz with 5\n\
            close @a when a.get(or: 5) == 42";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_invalid_is_fresh() {
        let spec = "input a: Int8\n\
        output b \n\
            eval @1Hz when a.is_fresh() with a.is_fresh()\n\
        output c(p) \n\
            spawn @1Hz with a.is_fresh() when a.is_fresh()\n\
            eval @1Hz with 5\n\
        output d \n\
            eval @1Hz with 5\n\
            close @0.5Hz when a.is_fresh()";
        assert_eq!(5, num_errors(spec));
    }

    #[test]
    fn test_valid_is_fresh() {
        let spec = "input a: Int8\n\
        output b \n\
            eval @a when a.is_fresh() with a.is_fresh()\n\
        output c(p) \n\
            spawn @a with a.is_fresh() when a.is_fresh()\n\
            eval @1Hz with 5\n\
        output d \n\
            eval @1Hz with 5\n\
            close @a when a.is_fresh()";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_discrete_window() {
        let spec = "input a: Int8\n\
        output b := a.aggregate(over_discrete: 5, using: sum)";

        assert_eq!(0, num_errors(spec));
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let b = tt[&NodeId::SRef(hir.outputs[0].sr)].clone();
        assert_eq!(
            b.eval_pacing,
            ConcretePacingType::Event(ActivationCondition::with_stream(hir.inputs[0].sr))
        );
    }

    #[test]
    fn test_discrete_window_periodic() {
        let spec = "input a: Int8\n\
        output b @1Hz := a.hold(or: 5)
        output c := b.aggregate(over_discrete: 5, using: sum)";

        assert_eq!(0, num_errors(spec));
        let (hir, _) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir);
        let tt = ltc.pacing_type_infer().unwrap();

        let c = tt[&NodeId::SRef(hir.outputs[1].sr)].clone();
        assert_eq!(
            c.eval_pacing,
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
    }

    #[test]
    fn test_disjunctive_annotated() {
        let spec = "input a: Int8\n\
        input b: Bool\n\
        output c @(a || b) := 42
        output d @a := c";

        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_disjunctive_annotated2() {
        let spec = "input a: Int8\n\
        input b: Bool\n\
        output c @(a || b) := 42
        output d @a&b := c";

        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_annotated_pacing_complex() {
        let spec = "input a: Int8\n\
        input b: Bool\n\
        input c: Bool\n\
        input d: Bool\n\
        output e @(((a&b)| c) & ((a & c) | b)) := 42
        output f @a&b := e";

        assert_eq!(0, num_errors(spec));
    }
}
