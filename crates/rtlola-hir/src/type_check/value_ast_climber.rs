use std::collections::HashMap;
use std::time::Duration;

use rtlola_reporting::{RtLolaError, Span};
use rusttyc::{TcErr, TcKey, TypeChecker, TypeTable};

use crate::hir::{
    AnnotatedType, Constant, Expression, ExpressionKind, FnExprKind, Hir, Inlined, Input, Literal, Offset, Output,
    SRef, StreamAccessKind, StreamReference, Trigger, WidenExprKind, WindowReference,
};
use crate::modes::HirMode;
use crate::type_check::pacing_types::Freq;
use crate::type_check::rtltc::{NodeId, TypeError};
use crate::type_check::value_types::{AbstractValueType, ValueErrorKind};
use crate::type_check::{ConcreteStreamPacing, ConcreteValueType};

/// A [Variable] is linked to a reusable [TcKey] in the RustTyc Type Checker.
/// e.g. used to reference stream-variables or parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Variable(String);

impl rusttyc::TcVar for Variable {}

impl Variable {
    /// Constructs the correct Variable for a Parameter, given the [Output] and the parameter `ìdx`.
    fn for_parameter(output: &Output, idx: usize) -> Self {
        Variable(output.name.clone() + "_" + &output.params[idx].name.clone())
    }
}

/// The [ValueTypeChecker] is used to infer the value type (e.g. Int32, Bool, Float64) for all expressions and streams.
pub(crate) struct ValueTypeChecker<'a, M>
where
    M: HirMode,
{
    /// The internal instance of a rusttyc [Typechecker].
    pub(crate) tyc: TypeChecker<AbstractValueType, Variable>,
    /// Maps a [NodeId] to a rusttyc TcKey. Allows referring to a TcKey for any given Hir node.
    pub(crate) node_key: HashMap<NodeId, TcKey>,
    /// Maps a TcKey to the Hir node span it represents. Used for error reporting.
    pub(crate) key_span: HashMap<TcKey, Span>,
    /// The input Hir.
    pub(crate) hir: &'a Hir<M>,
    /// The result of the pacing type analysis. Needed to determine the correct type of a realtime offset expression.
    pub(crate) pacing_tt: &'a HashMap<NodeId, ConcreteStreamPacing>,
    /// Storage to register exact type bounds during Hir climbing, resolved and checked during post process.
    pub(crate) annotated_checks: HashMap<TcKey, (ConcreteValueType, Option<TcKey>)>,
    /// Stores all widen checks during HIR climbing, resolved and checked during post process.
    pub(crate) widen_checks: HashMap<TcKey, (ConcreteValueType, TcKey)>,
    /// Lookup table for the name of a given stream.
    pub(crate) names: &'a HashMap<StreamReference, &'a str>,
}

impl<'a, M> ValueTypeChecker<'a, M>
where
    M: HirMode,
{
    /// Creates a new [ValueTypeChecker], requires a pacing type table given by [type_check](crate::type_check::PacingTypeChecker::type_check).
    /// `names` maps each stream reference to the name of the stream  referenced
    pub(crate) fn new(
        hir: &'a Hir<M>,
        names: &'a HashMap<StreamReference, &'a str>,
        pacing_tt: &'a HashMap<NodeId, ConcreteStreamPacing>,
    ) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = HashMap::new();
        let mut key_span = HashMap::new();
        let annotated_checks = HashMap::new();
        let widen_checks = HashMap::new();

        for input in hir.inputs() {
            let key = tyc.get_var_key(&Variable(input.name.clone()));
            node_key.insert(NodeId::SRef(input.sr), key);
            key_span.insert(key, input.span.clone());
        }

        for out in hir.outputs() {
            let key = tyc.get_var_key(&Variable(out.name.clone()));
            node_key.insert(NodeId::SRef(out.sr), key);
            key_span.insert(key, out.span.clone());
        }

        for (ix, tr) in hir.triggers().enumerate() {
            let n = format!("trigger_{}", ix);
            let key = tyc.get_var_key(&Variable(n));
            node_key.insert(NodeId::SRef(tr.sr), key);
            key_span.insert(key, tr.span.clone());
        }

        ValueTypeChecker {
            tyc,
            //decl,
            node_key,
            key_span,
            hir,
            pacing_tt,
            annotated_checks,
            widen_checks,
            names,
        }
    }

    pub(crate) fn type_check(mut self) -> Result<HashMap<NodeId, ConcreteValueType>, RtLolaError> {
        for input in self.hir.inputs() {
            self.input_infer(input)
                .map_err(|e| e.into_diagnostic(&[&self.key_span], &self.names))?;
        }

        for output in self.hir.outputs() {
            self.output_infer(output)
                .map_err(|e| e.into_diagnostic(&[&self.key_span], &self.names))?;
        }

        for trigger in self.hir.triggers() {
            self.trigger_infer(trigger)
                .map_err(|e| e.into_diagnostic(&[&self.key_span], &self.names))?;
        }

        let tt = self
            .tyc
            .clone()
            .type_check()
            .map_err(|e| TypeError::from(e).into_diagnostic(&[&self.key_span], &self.names))?;

        let mut error = RtLolaError::new();
        for err in Self::check_explicit_bounds(self.annotated_checks.clone(), &tt) {
            error.add(err.into_diagnostic(&[&self.key_span], &self.names));
        }

        for err in Self::check_widen_exprs(self.widen_checks.clone(), &tt) {
            error.add(err.into_diagnostic(&[&self.key_span], &self.names));
        }

        for err in Self::post_process(self.hir, &self.node_key, &tt) {
            error.add(err.into_diagnostic(&[&self.key_span], &self.names));
        }
        Result::from(error)?;

        let result_map = self
            .node_key
            .into_iter()
            .map(|(node, key)| (node, tt[&key].clone()))
            .collect();
        Ok(result_map)
    }

    fn match_const_literal(&self, lit: &Literal) -> AbstractValueType {
        match lit {
            Literal::Str(_) => AbstractValueType::String,
            Literal::Bool(_) => AbstractValueType::Bool,
            Literal::Integer(_) => AbstractValueType::Integer,
            Literal::SInt(_) => AbstractValueType::SInteger,
            Literal::Float(_) => AbstractValueType::Float,
        }
    }

    fn bind_to_annotated_type(
        &mut self,
        target: TcKey,
        bound: &AnnotatedType,
        conflict_key: Option<TcKey>,
    ) -> Result<(), TypeError<ValueErrorKind>> {
        let concrete_type = ConcreteValueType::from_annotated_type(bound).map_err(|reason| {
            TypeError {
                kind: reason,
                key1: Some(target),
                key2: None,
            }
        })?;
        self.annotated_checks.insert(target, (concrete_type, conflict_key));
        Ok(())
    }

    fn add_widen_check(
        &mut self,
        term_key: TcKey,
        inner_expr_key: TcKey,
        ty: &AnnotatedType,
    ) -> Result<(), TypeError<ValueErrorKind>> {
        let concrete_type = ConcreteValueType::from_annotated_type(ty).map_err(|reason| {
            TypeError {
                kind: reason,
                key1: Some(term_key),
                key2: None,
            }
        })?;
        self.widen_checks.insert(inner_expr_key, (concrete_type, term_key));
        Ok(())
    }

    fn concretizes_annotated_type(
        &mut self,
        target: TcKey,
        annotated_type: &AnnotatedType,
    ) -> Result<(), TypeError<ValueErrorKind>> {
        match annotated_type {
            AnnotatedType::String => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::String))?
            },
            AnnotatedType::Int(0) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::SInteger))?
            },
            AnnotatedType::Int(x) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::SizedSInteger(*x)))?
            },
            AnnotatedType::Float(0) => self.tyc.impose(target.concretizes_explicit(AbstractValueType::Float))?,
            AnnotatedType::Float(f) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::SizedFloat(*f)))?
            },
            AnnotatedType::UInt(0) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::UInteger))?
            },
            AnnotatedType::UInt(u) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::SizedUInteger(*u)))?
            },
            AnnotatedType::Bool => self.tyc.impose(target.concretizes_explicit(AbstractValueType::Bool))?,
            AnnotatedType::Bytes => self.tyc.impose(target.concretizes_explicit(AbstractValueType::Bytes))?,
            AnnotatedType::Option(op) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::Option))?;
                let child_key = self.tyc.get_child_key(target, 0)?;
                self.concretizes_annotated_type(child_key, op.as_ref())?
            },
            AnnotatedType::Tuple(children) => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::Tuple(children.len())))?;
                for (ix, child) in children.iter().enumerate() {
                    let child_key = self.tyc.get_child_key(target, ix)?;
                    self.concretizes_annotated_type(child_key, child)?;
                }
            },
            AnnotatedType::Numeric => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::Numeric))?
            },
            AnnotatedType::Sequence => {
                self.tyc
                    .impose(target.concretizes_explicit(AbstractValueType::Sequence))?
            },
            AnnotatedType::Param(_, _) => {
                unreachable!("Param-Type only reachable in function calls and Param-Output calls")
            },
        }
        Ok(())
    }

    fn handle_annotated_type(
        &mut self,
        target: TcKey,
        a_ty: &AnnotatedType,
        conflict: Option<TcKey>,
    ) -> Result<(), TypeError<ValueErrorKind>> {
        self.concretizes_annotated_type(target, a_ty)?;
        self.bind_to_annotated_type(target, a_ty, conflict)
    }

    /// Handles the annotated type for given [Input] stream.
    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<TcKey, TypeError<ValueErrorKind>> {
        let term_key: TcKey = *self
            .node_key
            .get(&NodeId::SRef(input.sr))
            .expect("Added in constructor");

        self.handle_annotated_type(term_key, &input.annotated_type, None)?;

        Ok(term_key)
    }

    /// Infers the type for a single [Output] stream.
    /// Performs check order:
    /// 1: Generates keys and checks for all parameters
    /// 2: Checks spawn condition and equates values with parameters
    /// 3: Analyse close and filter expressions
    /// 4: Check stream expression and sets expression type as stream value type
    pub(crate) fn output_infer(&mut self, out: &Output) -> Result<TcKey, TypeError<ValueErrorKind>> {
        let out_key = *self.node_key.get(&NodeId::SRef(out.sr)).expect("Added in constructor");

        let param_types: Vec<TcKey> = out
            .params
            .iter()
            .map(|param| {
                let param_key = self.tyc.get_var_key(&Variable::for_parameter(out, param.idx));

                self.node_key.insert(NodeId::Param(param.idx, out.sr), param_key);
                self.key_span.insert(param_key, param.span.clone());

                if let Some(a_ty) = param.annotated_type.as_ref() {
                    self.handle_annotated_type(param_key, a_ty, None)?;
                }
                Ok(param_key)
            })
            .collect::<Result<Vec<_>, TypeError<ValueErrorKind>>>()?;

        let opt_spawn = &self.hir.spawn(out.sr);
        if let Some((opt_spawn, opt_cond)) = opt_spawn {
            //check target expression type matches parameter type
            if let Some(spawn) = opt_spawn {
                let spawn_target_key = self.expression_infer(spawn, None)?;
                match param_types.len() {
                    0 => unreachable!("ensured by pacing type checker"),
                    1 => {
                        self.tyc.impose(spawn_target_key.equate_with(param_types[0]))?;
                    },
                    _ => {
                        self.tyc.impose(
                            spawn_target_key.concretizes_explicit(AbstractValueType::Tuple(param_types.len())),
                        )?;
                        let parameter_tuple = self.tyc.new_term_key();
                        for (ix, p) in param_types.iter().enumerate() {
                            let child = self.tyc.get_child_key(parameter_tuple, ix)?;
                            self.tyc.impose(child.equate_with(*p))?;
                        }
                        self.tyc.impose(parameter_tuple.equate_with(spawn_target_key))?;
                    },
                }
            }
            if let Some(cond) = opt_cond {
                self.expression_infer(cond, Some(AbstractValueType::Bool))?;
            }
        }
        if let Some(close) = &self.hir.close(out.sr) {
            self.expression_infer(close, Some(AbstractValueType::Bool))?;
        }
        if let Some(filter) = &self.hir.filter(out.sr) {
            self.expression_infer(filter, Some(AbstractValueType::Bool))?;
        }

        let expression_key = self.expression_infer(self.hir.expr(out.sr), None)?;
        if let Some(a_ty) = out.annotated_type.as_ref() {
            self.handle_annotated_type(out_key, a_ty, Some(expression_key))?;
        }

        self.tyc.impose(out_key.equate_with(expression_key))?;

        Ok(out_key)
    }

    /// Infers the type for a single [Trigger]. The trigger expression has to be of boolean type.
    pub(crate) fn trigger_infer(&mut self, tr: &Trigger) -> Result<TcKey, TypeError<ValueErrorKind>> {
        let tr_key = *self.node_key.get(&NodeId::SRef(tr.sr)).expect("added in constructor");
        let expression_key = self.expression_infer(&self.hir.expr(tr.sr), Some(AbstractValueType::Bool))?;
        self.tyc.impose(tr_key.concretizes(expression_key))?;
        Ok(tr_key)
    }

    /// Helper function for real time offsets.
    /// Checks if the offset is a multiple of the target stream frequency and try to convert it to a relative discrete offset.
    fn handle_realtime_offset(
        &mut self,
        target_ref: SRef,
        d: &Duration,
        term_key: TcKey,
        target_key: TcKey,
    ) -> Result<(), TypeError<ValueErrorKind>> {
        use num::rational::Rational64 as Rational;
        use uom::si::frequency::hertz;
        use uom::si::rational64::Frequency as UOM_Frequency;

        use crate::type_check::pacing_types::AbstractPacingType::*;
        let mut duration_as_f = d.as_secs_f64();
        let mut c = 0;
        while duration_as_f % 1.0f64 > 0f64 {
            c += 1;
            duration_as_f *= 10f64;
        }

        let rat = Rational::new(10i64.pow(c), duration_as_f as i64);
        let freq = Freq::Fixed(UOM_Frequency::new::<hertz>(rat));
        let target_ratio = self.pacing_tt[&NodeId::SRef(target_ref)]
            .expression_pacing
            .to_abstract_freq();
        //special case: period of current output > offset
        // && offset is multiple of target stream (no optional needed)
        if let Ok(Periodic(target_freq)) = target_ratio {
            //if the frequencies match the access is possible
            //dbg!(&freq, &target_freq);
            if let Ok(true) = target_freq.is_multiple_of(&freq) {
                //dbg!("frequencies compatible");
                self.tyc
                    .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                self.tyc.impose(target_key.equate_with(inner_key))?;
            } else {
                //dbg!("frequencies NOT compatible");
                //if the ey dont match return error
                return Err(TypeError {
                    kind: ValueErrorKind::IncompatibleRealTimeOffset(target_freq, duration_as_f as i64),
                    key1: Some(term_key),
                    key2: Some(target_key),
                });
            }
        } else {
            unreachable!("Ensured by pacing type checker!")
        }
        Ok(())
    }

    fn expression_infer(
        &mut self,
        exp: &Expression,
        target_type: Option<AbstractValueType>,
    ) -> Result<TcKey, TypeError<ValueErrorKind>> {
        let term_key: TcKey = self.tyc.new_term_key();
        self.node_key.insert(NodeId::Expr(exp.eid), term_key);
        self.key_span.insert(term_key, exp.span.clone());
        if let Some(t) = target_type {
            self.tyc.impose(term_key.concretizes_explicit(t))?;
        }

        match &exp.kind {
            ExpressionKind::LoadConstant(c) => {
                let cons_lit = match c {
                    Constant::Basic(lit) => lit,
                    Constant::Inlined(Inlined { lit, ty: anno_ty }) => {
                        self.handle_annotated_type(term_key, anno_ty, None)?;
                        lit
                    },
                };
                let literal_type = self.match_const_literal(cons_lit);
                self.tyc.impose(term_key.concretizes_explicit(literal_type))?;
            },

            ExpressionKind::StreamAccess(sr, kind, args) => {
                if sr.is_input() {
                    assert!(args.is_empty(), "parametrized input streams are unsupported");
                }

                if !args.is_empty() {
                    let target_stream: &Output = self.hir.output(*sr).expect("unable to find referenced stream");
                    let param_keys: Vec<_> = target_stream
                        .params
                        .iter()
                        .map(|p| {
                            // Todo: Move to own function in Variable
                            self.tyc.get_var_key(&Variable::for_parameter(target_stream, p.idx))
                        })
                        .collect();

                    let arg_keys: Vec<TcKey> = args
                        .iter()
                        .map(|arg| self.expression_infer(arg, None))
                        .collect::<Result<Vec<TcKey>, TypeError<ValueErrorKind>>>()?;

                    param_keys
                        .iter()
                        .zip(arg_keys.iter())
                        .map(|(p, a)| self.tyc.impose(a.equate_with(*p)))
                        .collect::<Result<Vec<()>, TcErr<AbstractValueType>>>()?;
                }

                let target_key = self.node_key.get(&NodeId::SRef(*sr)).expect("entered in constructor");

                match kind {
                    StreamAccessKind::Sync => {
                        self.tyc.impose(term_key.equate_with(*target_key))?;
                    },
                    StreamAccessKind::DiscreteWindow(wref) | StreamAccessKind::SlidingWindow(wref) => {
                        let (target, op, wait) = match wref {
                            WindowReference::Sliding(_) => {
                                let win = self.hir.single_sliding(*wref);
                                (win.target, win.aggr.op, win.aggr.wait)
                            },
                            WindowReference::Discrete(_) => {
                                let win = self.hir.single_discrete(*wref);
                                (win.target, win.aggr.op, win.aggr.wait)
                            },
                        };
                        let target_key = *self
                            .node_key
                            .get(&NodeId::SRef(target))
                            .expect("all nodes keys were entered in the constructor");

                        use rtlola_parser::ast::WindowOperation;
                        match op {
                            //Min|Max|Avg|Last|Percentile: <T:Num> T -> Option<T>
                            WindowOperation::Min
                            | WindowOperation::Max
                            | WindowOperation::Average
                            | WindowOperation::Last
                            | WindowOperation::NthPercentile(_) => {
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                self.tyc.impose(inner_key.equate_with(target_key))?;
                            },
                            //Count: Any -> uint
                            WindowOperation::Count => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Any))?;
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::UInteger))?;
                            },
                            //integral :T <T:Num> -> T
                            //integral : T <T:Num> -> Float   <-- currently used
                            WindowOperation::Integral => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Numeric))?;
                                if wait {
                                    self.tyc
                                        .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    //self.tyc.impose(inner_key.equate_with(ex_key))?;
                                    self.tyc
                                        .impose(inner_key.concretizes_explicit(AbstractValueType::Float))?;
                                } else {
                                    //self.tyc.impose(term_key.concretizes(ex_key))?;
                                    self.tyc
                                        .impose(term_key.concretizes_explicit(AbstractValueType::Float))?;
                                }
                            },
                            //Σ and Π :T <T:Num> -> T
                            WindowOperation::Sum | WindowOperation::Product => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Numeric))?;
                                if wait {
                                    self.tyc
                                        .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    self.tyc.impose(inner_key.concretizes(target_key))?;
                                } else {
                                    self.tyc.impose(term_key.concretizes(target_key))?;
                                }
                            },
                            //bool -> bool
                            WindowOperation::Conjunction | WindowOperation::Disjunction => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Bool))?;
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Bool))?;
                            },
                            // Float -> Option<Float>
                            WindowOperation::Variance | WindowOperation::StandardDeviation => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Float))?;
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                self.tyc.impose(inner_key.equate_with(target_key))?;
                            },
                            //<F:Float > (F,F) -> Option<F>
                            WindowOperation::Covariance => {
                                //Origin stream hs to be a tuple of size 2
                                self.tyc
                                    .impose(target_key.concretizes_explicit(AbstractValueType::Tuple(2)))?;
                                // Origin stream has to be a (Float, Float) tuple
                                let target_child_1 = self.tyc.get_child_key(target_key, 0)?;
                                let target_child_2 = self.tyc.get_child_key(target_key, 1)?;
                                self.tyc
                                    .impose(target_child_1.concretizes_explicit(AbstractValueType::Float))?;
                                self.tyc
                                    .impose(target_child_2.concretizes_explicit(AbstractValueType::Float))?;
                                //Result Key is option, window is failable (empty set)
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                //result inner key is float
                                self.tyc.impose(inner_key.is_meet_of(target_child_1, target_child_2))?;
                            },
                        }
                    },
                    StreamAccessKind::Hold => {
                        self.tyc
                            .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                        let inner_key = self.tyc.get_child_key(term_key, 0)?;
                        self.tyc.impose(target_key.equate_with(inner_key))?;
                    },
                    StreamAccessKind::Offset(off) => {
                        match off {
                            Offset::PastDiscrete(_) => {
                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Option))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                self.tyc.impose(target_key.equate_with(inner_key))?;
                            },
                            Offset::FutureRealTime(_) | Offset::FutureDiscrete(_) => {
                                panic!("future offsets are not supported")
                            },

                            Offset::PastRealTime(d) => {
                                debug_assert!(false, "real-time offsets are not supported yet");
                                let tk = *target_key;
                                self.handle_realtime_offset(*sr, d, term_key, tk)?;
                            },
                        }
                    },
                };
            },
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr, None)?; //Option<X>
                let def_key = self.expression_infer(&*default, None)?; // Y
                self.tyc
                    .impose(ex_key.concretizes_explicit(AbstractValueType::Option))?;
                let inner_key = self.tyc.get_child_key(ex_key, 0)?;
                self.tyc.impose(term_key.is_sym_meet_of(def_key, inner_key))?;
            },
            ExpressionKind::ArithLog(op, expr_v) => {
                use crate::hir::ArithLogOp;
                let arg_keys: Result<Vec<TcKey>, TypeError<ValueErrorKind>> =
                    expr_v.iter().map(|expr| self.expression_infer(expr, None)).collect();
                let arg_keys = arg_keys?;
                match arg_keys.len() {
                    2 => {
                        let left_key = arg_keys[0];
                        let right_key = arg_keys[1];
                        match op {
                            // <T:Num> T x T -> T
                            ArithLogOp::Add
                            | ArithLogOp::Sub
                            | ArithLogOp::Mul
                            | ArithLogOp::Div
                            | ArithLogOp::Rem
                            | ArithLogOp::Pow
                            | ArithLogOp::Shl
                            | ArithLogOp::Shr
                            | ArithLogOp::BitAnd
                            | ArithLogOp::BitOr
                            | ArithLogOp::BitXor => {
                                self.tyc
                                    .impose(left_key.concretizes_explicit(AbstractValueType::Numeric))?;
                                self.tyc
                                    .impose(right_key.concretizes_explicit(AbstractValueType::Numeric))?;

                                self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                                self.tyc.impose(term_key.equate_with(left_key))?;
                                self.tyc.impose(term_key.equate_with(right_key))?;
                            },
                            // Bool x Bool -> Bool
                            ArithLogOp::And | ArithLogOp::Or => {
                                self.tyc
                                    .impose(left_key.concretizes_explicit(AbstractValueType::Bool))?;
                                self.tyc
                                    .impose(right_key.concretizes_explicit(AbstractValueType::Bool))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Bool))?;
                            },
                            // Any x Any -> Bool COMPARATORS
                            ArithLogOp::Eq
                            | ArithLogOp::Lt
                            | ArithLogOp::Le
                            | ArithLogOp::Ne
                            | ArithLogOp::Ge
                            | ArithLogOp::Gt => {
                                self.tyc.impose(left_key.equate_with(right_key))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Bool))?;
                            },
                            ArithLogOp::Not | ArithLogOp::Neg | ArithLogOp::BitNot => {
                                unreachable!("unary operator cannot have 2 arguments")
                            },
                        }
                    },
                    1 => {
                        let arg_key = arg_keys[0];
                        match op {
                            // Bool -> Bool
                            ArithLogOp::Not => {
                                self.tyc.impose(arg_key.concretizes_explicit(AbstractValueType::Bool))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(AbstractValueType::Bool))?;
                            },
                            //Num -> Num
                            ArithLogOp::Neg | ArithLogOp::BitNot => {
                                self.tyc
                                    .impose(arg_key.concretizes_explicit(AbstractValueType::Numeric))?;

                                self.tyc.impose(term_key.equate_with(arg_key))?;
                            },
                            _ => unreachable!("All other operators have 2 given arguments"),
                        }
                    },
                    _ => unreachable!(),
                }
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                // Bool for condition - check given in the second argument
                self.expression_infer(&*condition, Some(AbstractValueType::Bool))?;
                let cons_key = self.expression_infer(&*consequence, None)?; // X
                let alt_key = self.expression_infer(&*alternative, None)?; // X
                                                                           //Bool x T x T -> T
                self.tyc.impose(term_key.is_sym_meet_of(cons_key, alt_key))?;
            },

            ExpressionKind::Tuple(vec) => {
                let key_vec = vec
                    .iter()
                    .map(|ex| self.expression_infer(ex, None))
                    .collect::<Result<Vec<TcKey>, TypeError<ValueErrorKind>>>()?;
                self.tyc
                    .impose(term_key.concretizes_explicit(AbstractValueType::Tuple(vec.len())))?;
                for (n, child_key_inferred) in key_vec.iter().enumerate() {
                    let n_key_given = self.tyc.get_child_key(term_key, n)?;
                    self.tyc.impose(n_key_given.equate_with(*child_key_inferred))?;
                }
            },

            ExpressionKind::TupleAccess(expr, idx) => {
                let ex_key = self.expression_infer(expr, None)?;
                self.tyc
                    .impose(ex_key.concretizes_explicit(AbstractValueType::AnyTuple))?;

                let accessed_child = self.tyc.get_child_key(ex_key, *idx)?;
                self.tyc.impose(term_key.equate_with(accessed_child))?;
            },

            ExpressionKind::Widen(WidenExprKind { expr: inner, ty }) => {
                let inner_expr_key = self.expression_infer(inner, None)?;
                // Bind expr type to least restrictive version of the variant i.e. e.g. Integer w/ width of 1 bit.
                // Also bind the result to the type annotation of the widening call.
                let upper_bound = match ty {
                    AnnotatedType::UInt(_) => AbstractValueType::UInteger,
                    AnnotatedType::Int(_) => AbstractValueType::SInteger,
                    AnnotatedType::Float(_) => AbstractValueType::Float,
                    _ => unimplemented!("unsupported widening type"),
                };
                self.handle_annotated_type(term_key, &ty, Some(inner_expr_key))?;
                self.tyc.impose(inner_expr_key.concretizes_explicit(upper_bound))?;
                self.add_widen_check(term_key, inner_expr_key, &ty)?;
                //self.tyc.impose(term_key.concretizes(inner_expr_key))?;
            },
            ExpressionKind::Function(FnExprKind { name, type_param, args }) => {
                // type_param.len() <= generics.len()
                // param.len() == args.len()

                // check for name in context

                // Function declaration:
                // cast<A, B>(x: A)          -> B
                //      ^  ^     ^              ^
                //      |  |     |              |
                //  Generics     parameter      result type

                // Function call:
                // cast<Int, Float>      (x)
                //       ^      ^         ^
                //       |      |         |
                //      type parameter    args

                let fun_decl = self.hir.func_declaration(name);
                //Generics
                if type_param.len() > fun_decl.generics.len() {
                    return Err(ValueErrorKind::UnnecessaryTypeParam(exp.span.clone()).into());
                }
                let generics: Vec<TcKey> = fun_decl
                    .generics
                    .iter()
                    .map(|gen| {
                        let gen_key: TcKey = self.tyc.new_term_key();
                        self.concretizes_annotated_type(gen_key, gen).map(|_| gen_key)
                    })
                    .collect::<Result<Vec<TcKey>, TypeError<ValueErrorKind>>>()?;

                // Bind generics to annotated types in function call
                for (t, gen) in type_param.iter().zip(generics.iter()) {
                    self.handle_annotated_type(*gen, t, None)?;
                }

                // function call arguments have type given by generics
                args.iter()
                    .zip(fun_decl.parameters.iter())
                    .map(|(arg, param)| {
                        // Replace reference to generic with generic key
                        let p = self.replace_type(param, &generics)?;
                        let arg_key = self.expression_infer(&*arg, None)?;
                        self.tyc.impose(arg_key.equate_with(p))?;
                        Ok(())
                    })
                    .collect::<Result<Vec<()>, TypeError<ValueErrorKind>>>()?;

                let return_type = self.replace_type(&fun_decl.return_type, &generics)?;

                self.tyc.impose(term_key.concretizes(return_type))?;
            },
            ExpressionKind::ParameterAccess(current_stream, ix) => {
                let output: &Output = self
                    .hir
                    .outputs()
                    .find(|o| o.sr == *current_stream)
                    .expect("Expect valid stream reference");
                let par_key = self.tyc.get_var_key(&Variable::for_parameter(output, *ix));
                //dbg!(par_key);
                self.tyc.impose(term_key.equate_with(par_key))?;
            },
        };

        Ok(term_key)
    }

    fn replace_type(&mut self, at: &AnnotatedType, to: &[TcKey]) -> Result<TcKey, TypeError<ValueErrorKind>> {
        match at {
            AnnotatedType::Param(idx, _) => Ok(to[*idx]),
            AnnotatedType::Numeric
            | AnnotatedType::Sequence
            | AnnotatedType::Int(_)
            | AnnotatedType::Float(_)
            | AnnotatedType::UInt(_)
            | AnnotatedType::Bool
            | AnnotatedType::String
            | AnnotatedType::Bytes
            | AnnotatedType::Option(_)
            | AnnotatedType::Tuple(_) => {
                let replace_key = self.tyc.new_term_key();
                self.concretizes_annotated_type(replace_key, at)?;
                Ok(replace_key)
            },
        }
    }

    pub(crate) fn check_explicit_bounds(
        annotated_checks: HashMap<TcKey, (ConcreteValueType, Option<TcKey>)>,
        tt: &TypeTable<AbstractValueType>,
    ) -> Vec<TypeError<ValueErrorKind>> {
        annotated_checks
            .into_iter()
            .filter_map(|(target, (bound, conflict))| {
                let resolved = tt[&target].clone();
                if resolved != bound {
                    Some(TypeError {
                        kind: ValueErrorKind::ExactTypeMismatch(resolved, bound),
                        key1: Some(target),
                        key2: conflict,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    pub(crate) fn check_widen_exprs(
        widen_checks: HashMap<TcKey, (ConcreteValueType, TcKey)>,
        tt: &TypeTable<AbstractValueType>,
    ) -> Vec<TypeError<ValueErrorKind>> {
        widen_checks
            .into_iter()
            .filter_map(|(inner_key, (bound, parent))| {
                let resolved = tt[&inner_key].clone();
                match (bound.width(), resolved.width()) {
                    (Some(bound_width), Some(inner_width)) if inner_width > bound_width => {
                        Some(TypeError {
                            kind: ValueErrorKind::InvalidWiden(bound, resolved),
                            key1: Some(parent),
                            key2: Some(inner_key),
                        })
                    },
                    _ => None,
                }
            })
            .collect()
    }

    pub(crate) fn post_process(
        hir: &Hir<M>,
        node_key: &HashMap<NodeId, TcKey>,
        tt: &TypeTable<AbstractValueType>,
    ) -> Vec<TypeError<ValueErrorKind>> {
        let mut errors = vec![];
        //Check that no output or spawn target has an optional type
        for output in &hir.outputs {
            let key = node_key[&NodeId::SRef(output.sr())];
            let ty: &ConcreteValueType = &tt[&key];
            if matches!(ty, ConcreteValueType::Option(_)) {
                errors.push(TypeError {
                    kind: ValueErrorKind::OptionNotAllowed(ty.clone()),
                    key1: Some(key),
                    key2: None,
                });
            }

            if let Some(target) = output.instance_template.spawn.as_ref().and_then(|st| st.target) {
                let key = node_key[&NodeId::Expr(target)];
                let ty: &ConcreteValueType = &tt[&key];
                match ty {
                    ConcreteValueType::Tuple(child_types) => {
                        if let ExpressionKind::Tuple(children) = &hir.expression(target).kind {
                            if let Some((child_idx, child_ty)) = child_types
                                .iter()
                                .find_position(|t| matches!(t, ConcreteValueType::Option(_)))
                            {
                                let key = node_key[&NodeId::Expr(children[child_idx].eid)];
                                errors.push(TypeError {
                                    kind: ValueErrorKind::OptionNotAllowed(child_ty.clone()),
                                    key1: Some(key),
                                    key2: None,
                                })
                            }
                        }
                    },
                    ConcreteValueType::Option(_) => {
                        errors.push(TypeError {
                            kind: ValueErrorKind::OptionNotAllowed(ty.clone()),
                            key1: Some(key),
                            key2: None,
                        })
                    },
                    _ => {},
                }
                if matches!(ty, ConcreteValueType::Option(_)) {}
            }
        }
        errors
    }
}

#[cfg(test)]
mod value_type_tests {
    use std::collections::HashMap;

    use rtlola_parser::ast::RtLolaAst;
    use rtlola_parser::{parse, ParserConfig};

    use crate::hir::{ExprId, RtLolaHir, StreamReference};
    use crate::modes::BaseMode;
    use crate::type_check::rtltc::{LolaTypeChecker, NodeId};
    use crate::type_check::ConcreteValueType;

    struct TestBox {
        hir: RtLolaHir<BaseMode>,
    }

    impl TestBox {
        fn output(&self, name: &str) -> StreamReference {
            self.hir.get_output_with_name(name).unwrap().sr
        }

        fn input(&self, name: &str) -> StreamReference {
            self.hir.get_input_with_name(name).unwrap().sr
        }
    }

    fn setup_hir(spec: &str) -> TestBox {
        let ast: RtLolaAst = match parse(ParserConfig::for_string(spec.to_string())) {
            Ok(s) => s,
            Err(e) => panic!("Spec {} cannot be parsed: {:?}", spec, e),
        };
        let hir = crate::from_ast(ast).unwrap();
        //let mut dec = na.check(&spec);
        TestBox { hir }
    }

    fn check_value_type(spec: &str) -> (TestBox, HashMap<NodeId, ConcreteValueType>) {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir);
        let pacing_tt = ltc.pacing_type_infer().expect("Expected valid pacing type");
        let tt_result = ltc.value_type_infer(&pacing_tt);
        let tt = tt_result.expect("Expect Valid Input - Value Type check failed");
        (test_box, tt)
    }

    fn num_errors(spec: &str) -> usize {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir);
        let pt = ltc.pacing_type_infer().expect("expect valid pacing input");
        match ltc.value_type_infer(&pt) {
            Ok(_) => 0,
            Err(e) => e.num_errors(),
        }
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn direct_implication() {
        let spec = "input i: Int8\noutput o := i";
        let (tb, result_map) = check_value_type(spec);
        let input_sr = tb.hir.get_input_with_name("i").unwrap().sr;
        let output_sr = tb.hir.get_output_with_name("o").unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_sr)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_sr)], ConcreteValueType::Integer8);
    }

    #[test]
    fn direct_widening() {
        let spec = "input i: Int8\noutput o :Int32 := widen<Int32>(i)";
        let (tb, result_map) = check_value_type(spec);
        let input_sr = tb.hir.get_input_with_name("i").unwrap().sr;
        let output_sr = tb.hir.get_output_with_name("o").unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_sr)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_sr)], ConcreteValueType::Integer32);
    }

    #[test]
    fn integer_addition_wideing() {
        let spec = "input i: Int8\ninput i1: Int16\noutput o := widen<Int16>(i) + i1";
        let (tb, result_map) = check_value_type(spec);
        let mut input_iter = tb.hir.inputs();
        let input_i_id = input_iter.next().unwrap().sr;
        let input_i1_id = input_iter.next().unwrap().sr;
        let output_id = tb.hir.outputs().next().unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_i_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(input_i1_id)], ConcreteValueType::Integer16);
        assert_eq!(result_map[&NodeId::SRef(output_id)], ConcreteValueType::Integer16);
    }

    #[test]
    fn parametric_access_default() {
        let spec = "output i(a: Int8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true):= if b then a else 0 \n output o(x) spawn @1Hz with 42 := i(5,true).offset(by:-1).defaults(to: 42)";
        let (tb, result_map) = check_value_type(spec);
        let o2_sr = tb.output("i");
        let o1_id = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(o1_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(o2_sr)], ConcreteValueType::Integer8);
    }

    #[test]
    fn parametric_declaration_x() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true) := 1";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(result_map[&NodeId::SRef(output_sr)], ConcreteValueType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_infer() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz spawn @1Hz with (5, true) := a";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(result_map[&NodeId::SRef(output_sr)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true) := 1 output y @1Hz := x(1, false).hold().defaults(to:5)";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.output("x");
        let output_2_id = tb.output("y");
        assert_eq!(result_map[&NodeId::SRef(output_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_2_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_two_many() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz spawn @1Hz with (5, true, false) := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn parametric_declaration_param_two_few() {
        let spec = "output x(a: UInt8, b: Bool, c:String) @1Hz spawn @1Hz with (5, true) := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn parametric_declaration_param_infer2() {
        let spec = "output x (a, b) @1Hz spawn @1Hz with (5, true) := 42";
        let (tb, result_map) = check_value_type(spec);
        let x = tb.output("x");
        assert_eq!(result_map[&NodeId::Param(0, x)], ConcreteValueType::Integer64);
        assert_eq!(result_map[&NodeId::Param(1, x)], ConcreteValueType::Bool);
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(sr)], ConcreteValueType::Float32);
    }

    #[test]
    //TODO float16 annotation is treaded as Floa32, as Float16 is not implemented as Concrete Type
    fn simple_const_float16() {
        let spec = "constant c: Float16 := 2.1 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(sr)], ConcreteValueType::Float32);
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(sr)], ConcreteValueType::Integer8);
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true output o @1Hz := c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2 output o @1Hz := c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3 output o @1Hz := c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn simple_valid_coersion() {
        //TODO does not check output type, only validity
        for spec in &[
            "constant c: Int8 := 1\noutput o: Int32 @1Hz:= widen<Int32>(c)",
            "constant c: UInt16 := 1\noutput o: UInt64 @1Hz:= widen<UInt64>(c)",
            "constant c: Float32 := 1.0\noutput o: Float64 @1Hz:= widen<Float64>(c)",
        ] {
            assert_eq!(0, num_errors(spec));
        }
    }

    #[test]
    fn simple_invalid_conversion() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 @1Hz := c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn simple_invalid_widening() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 @1Hz := widen<Int8>(c)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn simple_explicit_widening() {
        let spec = "constant c: Int32 := 1\n constant d: Int8 := 2\noutput o @1Hz := c + widen<Int32>(d)";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(sr)], ConcreteValueType::Integer32);
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger @1Hz false";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.hir.triggers().nth(0).unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(tr_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn simple_trigger_message() {
        let spec = "trigger @1Hz false \"alert always\"";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.hir.triggers().nth(0).unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(tr_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn faulty_trigger() {
        let spec = "trigger @1Hz 1";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 @1Hz := 3 + 5";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.hir.outputs().next().unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn simple_binary_input() {
        let spec = "input i: Int8\noutput o: Int8 := 3 + i";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let in_id = tb.input("i");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn simple_unary() {
        let spec = "output o @1Hz:= !false \n\
                               output u: Bool @1Hz:= !false";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let out_id_2 = tb.output("u");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
        assert_eq!(result_map[&NodeId::SRef(out_id_2)], ConcreteValueType::Bool);
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool @1Hz:= !3";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_boolean_operator() {
        let spec = "
input i_0: Bool
output o_0: Bool @i_0 := !false
output o_1: Bool @i_0 := !true
output o_2: Bool @i_0 := false ∨ false
output o_3: Bool @i_0 := false ∨ true
output o_4: Bool @i_0 := true  || false
output o_5: Bool @i_0 := true  || true
output o_6: Bool @i_0 := false ∧ false
output o_7: Bool @i_0 := false ∧ true
output o_8: Bool @i_0 := true  && false
output o_9: Bool @i_0 := true  && true";
        let (tb, result_map) = check_value_type(spec);
        for i in 0..10 {
            let out_id = tb.output(&format!("o_{}", i));
            assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
        }

        let in_id = tb.input("i_0");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 @1Hz := false + 2.5";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 @1Hz := if false then 1 else 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn simple_ite_compare() {
        let spec = "output e :Int8 @1Hz := if 1 == 0 then 0 else -1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("e");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o @1Hz := if !false then 1.3 else -2.0";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 @1Hz := if 3 then 1 else 1";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 @1Hz := if true then 1 else false";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_parenthesized_expr() {
        let spec = "input s: String\noutput o: Bool := s[-1].defaults(to: \"\") == \"a\"";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let in_id = tb.input("s");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::TString);
    }

    #[test]
    fn test_underspecified_type() {
        //Default for num literals applied
        let spec = "output o @1Hz := 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let res_type = &result_map[&NodeId::SRef(out_id)];
        assert!(*res_type == ConcreteValueType::Integer32 || *res_type == ConcreteValueType::Integer64);
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        let in_id = tb.input("a");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 @1Hz:= 3\n output b: UInt8 := a[0]";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 @1Hz := 3\n output b: UInt8 := a[-1].defaults(to: 3)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        let in_id = tb.output("a");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_offset_regression() {
        let spec = "input a: UInt8 \n output sum := sum[-1].defaults(to: 0) + a";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("a");
        let out_id = tb.output("sum");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 @1Hz:= 3\n output b: Bool := a[-1].defaults(to: false)";
        assert_eq!(1, num_errors(spec));
    }
    #[test]
    //#[ignore] // paramertic streams need new design after syntax revision
    fn test_filter_type() {
        let spec = "input in: Bool\n output a: Int8 filter in := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Bool);
    }

    #[test]
    //#[ignore] // paramertic streams need new design after syntax revision
    fn test_filter_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 filter in := 3";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_close_type() {
        let spec = "input in: Bool\n output a: Int8 @1Hz close in := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn test_close_type_faulty() {
        //Close condition non boolean type
        let spec = "input in: Int8\n output a: Int8 @1Hz close in := 3";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_param_spec() {
        let spec =
            "output a(p1: Int8): Int8 @1Hz spawn @1Hz with 3 := 3\noutput b(p:Int8): Int8 spawn @1Hz with 42 := a(3)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::Integer8);
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "output a(p1: Int8): Int8 @1Hz spawn @1Hz with false := 3";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_param_inferred() {
        let spec = "input i: Int8\noutput x(param): Int8 spawn with i := i";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("x");
        let in_id = tb.input("i");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn test_param_inferred_conflicting() {
        let spec = "input i: Int8\ninput j: UInt8\noutput x(param): Int8 spawn @1Hz with 3 := i\noutput y: Int8 := x(i).hold().defaults(to: 42)\noutput z: Int8 := x(j).hold().defaults(to: 42)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_lookup_incomp() {
        let spec =
            "output a(p1: Int8): Int8 @1Hz spawn @1Hz with 42 := 3\n output b: UInt8 @1Hz := a(3).hold().defaults(to:3)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) @1Hz:= (14, false)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        assert_eq!(
            result_map[&NodeId::SRef(out_id)],
            ConcreteValueType::Tuple(vec![ConcreteValueType::Integer8, ConcreteValueType::Bool])
        );
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) @1Hz := (14, 3)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        assert_eq!(
            result_map[&NodeId::SRef(in_id)],
            ConcreteValueType::Tuple(vec![ConcreteValueType::Integer8, ConcreteValueType::Bool])
        );
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn test_tuple_access_faulty_type() {
        //TODO optional result at offset zero
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].0";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_tuple_access_faulty_len() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.2";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_optional_type() {
        let spec = "input in: Int8\noutput out: Int8 := in.offset(by: -1).defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(
            result_map[&NodeId::Expr(ExprId(1))],
            ConcreteValueType::Option(ConcreteValueType::Integer8.into())
        );
    }

    #[test]
    fn test_optional_type_faulty() {
        let spec = "input in: Int8\noutput out: Int8? := in";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    #[ignore] // Future offsets are not yet supported
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("a");
        let out_id = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_tuple_of_tuples() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Int16 := widen<Int16>(in[0].0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        let input_type = ConcreteValueType::Tuple(vec![
            ConcreteValueType::Integer8,
            ConcreteValueType::Tuple(vec![ConcreteValueType::UInteger8, ConcreteValueType::Bool]),
        ]);
        assert_eq!(result_map[&NodeId::SRef(in_id)], input_type);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer16);
    }

    #[test]
    fn test_tuple_of_tuples2() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Bool := in.1.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        let input_type = ConcreteValueType::Tuple(vec![
            ConcreteValueType::Integer8,
            ConcreteValueType::Tuple(vec![ConcreteValueType::UInteger8, ConcreteValueType::Bool]),
        ]);
        assert_eq!(result_map[&NodeId::SRef(in_id)], input_type);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn test_tuple_of_tuples3() {
        let spec = "output out: Bool @1Hz := (5, (7.0, true)).1.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Bool);
    }

    #[test]
    fn test_tuple_of_tuples4() {
        let spec = "output out1 := out2.1.1\noutput out2 @1Hz := (5, (7.0, true))";
        let (tb, result_map) = check_value_type(spec);
        let out2 = tb.output("out2");
        let out1 = tb.output("out1");
        let tuple_type = ConcreteValueType::Tuple(vec![
            ConcreteValueType::Integer64,
            ConcreteValueType::Tuple(vec![ConcreteValueType::Float32, ConcreteValueType::Bool]),
        ]);
        assert_eq!(result_map[&NodeId::SRef(out2)], tuple_type);
        assert_eq!(result_map[&NodeId::SRef(out1)], ConcreteValueType::Bool);
    }

    #[test]
    fn test_faulty_option_access() {
        let spec = "input x:Int32\noutput out @1Hz := x.hold().0";
        assert_ne!(0, num_errors(spec));
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 @5Hz:= widen<Int64>(in.aggregate(over: 3s, using: Σ))";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer64);
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    #[ignore] // uint to int cast currently not allowed
    fn test_aggregation_implicit_cast() {
        let spec = "input in: UInt8\n\
             output out: Int16 @5Hz := in.aggregate(over_exactly: 3s, using: Σ).defaults(to: 5)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer16);
    }

    #[test]
    #[ignore] //int to float cast currently not allowed
    fn test_aggregation_implicit_cast2() {
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: avg).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    //#[ignore] //symmetric type relation extends input int8 to float
    fn test_aggregation_implicit_cast3() {
        let spec =
                            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_aggregation_integer_integral() {
        let spec =
                            "input in: UInt8\n output out: UInt8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        assert_eq!(1, num_errors(spec));
        let spec =
            "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        assert_eq!(1, num_errors(spec));
        let spec =
            "input in: UInt8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
        let spec =
            "input in: Int8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_new_aggr_function_last() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: last).defaults(to: 4)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
    }

    #[test]
    fn test_new_aggr_function_med() {
        let spec = "input in: UInt8\n output out: UInt8 @5Hz := in.aggregate(over: 3s, using: µ).defaults(to: 4)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_new_aggr_function_percentile() {
        let spec = "input in: UInt8\n output out: UInt8 @5Hz := in.aggregate(over: 3s, using: pctl25).defaults(to: 4)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_new_aggr_function_variance() {
        let spec =
            "input in: Float32\n output out: Float32 @5Hz := in.aggregate(over: 3s, using: var).defaults(to: 0.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Float32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_new_aggr_function_covariance() {
        let spec = "input in: Float32\n input in2: Float32\noutput t:= (in,in2)\n output out: Float32 @5Hz := t.aggregate(over: 3s, using: cov).defaults(to: 0.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let in2_id = tb.input("in2");
        let t_id = tb.output("t");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Float32);
        assert_eq!(result_map[&NodeId::SRef(in2_id)], ConcreteValueType::Float32);
        assert_eq!(
            result_map[&NodeId::SRef(t_id)],
            ConcreteValueType::Tuple(vec![ConcreteValueType::Float32, ConcreteValueType::Float32])
        );
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_cov_result_type_check() {
        let spec = "input in: Float32\n input in2: Float64\noutput t:= (in,in2)\n output out: Float32 @5Hz := t.aggregate(over: 3s, using: covariance).defaults(to: 0.0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_cov_int() {
        let spec = "input in: Float32\n input in2: Int64\noutput t:= (in,in2)\n output out: Float64 @5Hz := t.aggregate(over: 3s, using: cov).defaults(to: 0.0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_new_aggr_function_sd() {
        let spec = "input in: Float32\n output out: Float32 @5Hz := in.aggregate(over: 3s, using: σ).defaults(to: 0.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Float32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_new_aggr_function_missing_default() {
        let spec = "input in: Float32\n output out: Float64 @5Hz := in.aggregate(over: 3s, using: σ²)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := widen<Float64>(velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0))";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("velo");
        let out_id = tb.output("avg");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Float32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float64);
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_rt_offset() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to:1)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::Integer8);
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_rt_offset_regression() {
        let spec = "output a @10Hz := a.offset(by: -100ms).defaults(to:0) + 1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer32);
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_rt_offset_regression2() {
        let spec = "
                            output x @ 10Hz := 1
                            output x_diff := x - x.offset(by:-1s).defaults(to: x)
                        ";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("x");
        let out_id2 = tb.output("x_diff");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer32);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::Integer32);
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-1s].defaults(to:1)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::Integer8);
    }

    #[test]
    #[ignore] // Real time offsets are not supported yet
    fn test_rt_offset_skip2() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-2s].defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], ConcreteValueType::Integer8);
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger8);
    }

    #[test]
    #[ignore] //implicit casting not usable currently
    fn test_casting_implicit_types() {
        let spec = "input x: UInt8\noutput y: Float32 := cast(x)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_casting_explicit_types() {
        let spec = "input x: Int32\noutput y: UInt32 := cast<Int32,UInt32>(x)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(result_map[&NodeId::SRef(in_id)], ConcreteValueType::Integer32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::UInteger32);
    }

    #[test]
    fn infinite_recursion_regression() {
        // this should fail in type checking as the value type of `c` cannot be determined.
        // it currently fails because default expects a optional value
        let spec = "output c @1Hz := c.defaults(to:0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_simple_annotated() {
        let spec = "output c: Int8 @1Hz := 42";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_simple_annotated2() {
        let spec = "output c: Int8 @1Hz := widen<Int32>(42)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn function_to_many_type_args() {
        let spec = "import math output c: Int32 @1Hz := max<Int16,Int16>(13,42)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn math_function_wrong_arg_type() {
        let spec = "import math output c @1Hz := sqrt(13.37) + cos(13)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_matches() {
        let spec = "import regex\ninput s: String\ninput b: Bytes\noutput c: Bool := s.matches<String>(regex:\"hello\") && b.matches<Bytes>(regex:\"world\")";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_max() {
        let spec = "import math\ninput s: Int16\ninput b: Int16\noutput c := max(s, b)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("c");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Integer16);
    }

    #[test]
    fn test_error_msg_bounds() {
        let spec = "input a: UInt8\noutput b: UInt16 := if true then a else a[-1].defaults(to: 0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_error_msg_bounds2() {
        let spec = "output a: UInt8 @1Hz := 5\noutput b: UInt16 := if true then a else a[-1].defaults(to: 0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_error_faulty_widen() {
        let spec = "input a: Int32\noutput b: Int16 := widen<Int16>(a)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sqrt() {
        let spec = "import math\ninput a: Float32\noutput b := sqrt(a)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_generics() {
        let spec = "import math\nconstant x: Float32 := 5.0\noutput b @1Hz := abs(x)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        assert_eq!(result_map[&NodeId::SRef(out_id)], ConcreteValueType::Float32);
    }

    #[test]
    fn test_complex_spec() {
        let spec = "\
        import math\n\
        input w_dir: Float64\n\
        input lat: Float64\n\
        input lon: Float64\n\
        output lon_diff: Float64 := lon - lon.offset(by: -1).defaults(to: lon)\n\
        output lat_diff: Float64 := lat - lat.offset(by: -1).defaults(to: lat)\n\
        output yaw: Float64 := if lon_diff = 0.0 then 0.0 else arctan(lat_diff / lon_diff)\n\
        output head_wind: Bool @ 1Hz := (w_dir.hold().defaults(to: 0.0) - yaw.hold().defaults(to: 0.0)) < 0.2";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_stdlib_function() {
        let spec = "import math\n\
                        input  a : Float64\n\
                        output b : Float64 := arctan(a)\n\
                        output c : Float64 := sin(a)\n\
                        ";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_no_optional_stream() {
        let spec = "input  a : Float64\n\
                         output b := a.offset(by:-1)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_no_optional_spawn_target() {
        let spec = "input  a : Float64\n\
                         output b(p) spawn with a.offset(by: -1) := a + p.defaults(to: 0.0)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_no_optional_spawn_target_tuple() {
        let spec = "input  a : Float64\n\
                         output b(p, q) spawn with (42, a.offset(by: -1)) := a + q.defaults(to: 0.0)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }
}
