extern crate regex;

use super::rusttyc::TypeTable;
use crate::common_ir::{Offset, StreamAccessKind};
use crate::hir::expression::{Constant, ConstantLiteral, Expression, ExpressionKind};
use crate::hir::modes::ir_expr::WithIrExpr;
use crate::hir::modes::HirMode;
use crate::hir::{AnnotatedType, Input, Output, Trigger, Window};
use crate::reporting::{Handler, Span};
use crate::tyc::pacing_types::ConcreteStreamPacing;
use crate::tyc::value_types::IConcreteType;
use crate::tyc::{pacing_types::Freq, rtltc::NodeId, value_types::IAbstractType};
use crate::RTLolaHIR;
use bimap::BiMap;
use itertools::Either;
use rusttyc::{TcErr, TcKey, TypeChecker};
use std::collections::HashMap;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct ValueContext<'a, M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub(crate) tyc: TypeChecker<IAbstractType, Variable>,
    //pub(crate) decl: DeclarationTable,
    //Map assumes uniqueness of Ast and Tc ids
    pub(crate) node_key: BiMap<NodeId, TcKey>,
    pub(crate) key_span: HashMap<TcKey, Span>,
    pub(crate) handler: &'a Handler,
    pub(crate) hir: &'a RTLolaHIR<M>,
    pub(crate) pacing_tt: &'a HashMap<NodeId, ConcreteStreamPacing>,
    pub(crate) annotated_checks: HashMap<TcKey, (IConcreteType, Option<TcKey>)>,
}

impl<'a, M> ValueContext<'a, M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub fn new(
        hir: &'a RTLolaHIR<M>,
        //decl: DeclarationTable,
        handler: &'a Handler,
        pacing_tt: &'a HashMap<NodeId, ConcreteStreamPacing>,
    ) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = BiMap::new();
        let key_span = HashMap::new();
        let annotated_checks = HashMap::new();

        for input in hir.inputs() {
            let key = tyc.get_var_key(&Variable { name: input.name.clone() });
            node_key.insert(NodeId::SRef(input.sr), key);
            //key_span.insert(key, input.span);
        }

        for out in hir.outputs() {
            let key = tyc.get_var_key(&Variable { name: out.name.clone() });
            node_key.insert(NodeId::SRef(out.sr), key);
            //key_span.insert(key, out.span);
        }

        for (ix, tr) in hir.triggers().enumerate() {
            let n = format!("trigger_{}", ix);
            let key = tyc.get_var_key(&Variable { name: n });
            node_key.insert(NodeId::SRef(tr.sr), key);
            //key_span.insert(key, tr.span);
        }

        ValueContext {
            tyc,
            //decl,
            node_key,
            key_span,
            handler,
            hir,
            pacing_tt,
            annotated_checks,
        }
    }

    fn match_const_literal(&self, lit: &ConstantLiteral) -> IAbstractType {
        match lit {
            ConstantLiteral::Str(_) => IAbstractType::TString,
            ConstantLiteral::Bool(_) => IAbstractType::Bool,
            ConstantLiteral::Integer(_) => IAbstractType::Integer,
            ConstantLiteral::SInt(_) => IAbstractType::SInteger(1),
            ConstantLiteral::Float(_) => IAbstractType::Float(1),
        }
    }

    pub fn bind_to_annotated_type(
        &mut self,
        target: TcKey,
        bound: &AnnotatedType,
        conflict_key: Option<TcKey>,
    ) -> Result<(), TcErr<IAbstractType>> {
        let concrete_type =
            IConcreteType::from_annotated_type(bound).map_err(|reason| TcErr::Bound(target, None, reason))?;
        self.annotated_checks.insert(target, (concrete_type, conflict_key));
        Ok(())
    }

    pub fn concretizes_annotated_type(
        &mut self,
        target: TcKey,
        annotated_type: &AnnotatedType,
    ) -> Result<(), TcErr<IAbstractType>> {
        match annotated_type {
            AnnotatedType::String => self.tyc.impose(target.concretizes_explicit(IAbstractType::TString)),
            AnnotatedType::Int(x) => self.tyc.impose(target.concretizes_explicit(IAbstractType::SInteger(*x))),
            AnnotatedType::Float(f) => self.tyc.impose(target.concretizes_explicit(IAbstractType::Float(*f))),
            AnnotatedType::UInt(u) => self.tyc.impose(target.concretizes_explicit(IAbstractType::UInteger(*u))),
            AnnotatedType::Bool => self.tyc.impose(target.concretizes_explicit(IAbstractType::Bool)),
            AnnotatedType::Bytes => self.tyc.impose(target.concretizes_explicit(IAbstractType::Bytes)),
            AnnotatedType::Option(op) => {
                self.tyc.impose(target.concretizes_explicit(IAbstractType::Option))?;
                let child_key = self.tyc.get_child_key(target, 0)?;
                self.concretizes_annotated_type(child_key, op.as_ref())
            }
            AnnotatedType::Tuple(children) => {
                self.tyc.impose(target.concretizes_explicit(IAbstractType::Tuple(children.len())))?;
                for (ix, child) in children.iter().enumerate() {
                    let child_key = self.tyc.get_child_key(target, ix)?;
                    self.concretizes_annotated_type(child_key, child)?;
                }
                Ok(())
            }
            AnnotatedType::Numeric => self.tyc.impose(target.concretizes_explicit(IAbstractType::Numeric)),
            AnnotatedType::Param(_, _) => {
                unreachable!("Param-Type only reachable in function calls and Param-Output calls")
            }
        }
    }

    pub fn input_infer(&mut self, input: &Input) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = *self.node_key.get_by_left(&NodeId::SRef(input.sr)).expect("Added in constructor");
        //Annotated Type

        self.concretizes_annotated_type(term_key, &input.annotated_type)?;
        self.bind_to_annotated_type(term_key, &input.annotated_type, None)?;

        /*
        let mut param_types = Vec::new();

        for param in &input.params {
            let param_key = self.tyc.get_var_key(&Variable {
                name: param.name.name.clone(),
            });
            dbg!(param_key);
            let t = self.type_kind_match(&param.ty);
            self.tyc.impose(param_key.concretizes_explicit(t))?;
            param_types.push(param_key);
        }
        assert!(
            param_types.is_empty(),
            "parametric input types currently not supported"
        );
        */
        Ok(term_key)
    }

    pub fn output_infer(&mut self, out: &Output) -> Result<TcKey, TcErr<IAbstractType>> {
        let out_key = *self.node_key.get_by_left(&NodeId::SRef(out.sr)).expect("Added in constructor");

        let mut param_types = Vec::new();
        for param in &out.params {
            let param_key = self.tyc.get_var_key(&Variable { name: out.name.clone() + "_" + &param.name.clone() });

            self.node_key.insert(NodeId::Param(param.idx, out.sr), param_key);
            self.key_span.insert(param_key, param.span.clone());

            if let Some(a_ty) = param.annotated_type.as_ref() {
                self.concretizes_annotated_type(param_key, a_ty)?;
                self.bind_to_annotated_type(param_key, a_ty, None)?;
            }
            param_types.push(param_key);
        }

        //dbg!(&out.instance_template);
        let opt_spawn = &self.hir.spawn(out.sr);
        if let Some((opt_spawn, opt_cond)) = opt_spawn {
            //chek target exression type matches parameter type
            if let Some(spawn) = opt_spawn {
                let spawn_target_key = self.expression_infer(spawn, None)?;
                match param_types.len() {
                    0 => return Err(TcErr::Bound(spawn_target_key, None, "Spawn condition without parameters".into())),
                    1 => {
                        self.tyc.impose(spawn_target_key.equate_with(param_types[0]))?;
                    }
                    _ => {
                        self.tyc
                            .impose(spawn_target_key.concretizes_explicit(IAbstractType::Tuple(param_types.len())))?;
                        let parameter_tuple = self.tyc.new_term_key();
                        for (ix, p) in param_types.iter().enumerate() {
                            let child = self.tyc.get_child_key(parameter_tuple, ix)?;
                            self.tyc.impose(child.equate_with(*p))?;
                        }
                        self.tyc.impose(parameter_tuple.equate_with(spawn_target_key))?;
                    }
                }
            }
            if let Some(cond) = opt_cond {
                self.expression_infer(cond, Some(IAbstractType::Bool))?;
            }
        }
        if let Some(close) = &self.hir.close(out.sr) {
            self.expression_infer(close, Some(IAbstractType::Bool))?;
        }
        if let Some(filter) = &self.hir.filter(out.sr) {
            self.expression_infer(filter, Some(IAbstractType::Bool))?;
        }

        let expression_key = self.expression_infer(self.hir.expr(out.sr), None)?;
        if let Some(a_ty) = out.annotated_type.as_ref() {
            self.concretizes_annotated_type(out_key, a_ty)?;
            self.bind_to_annotated_type(out_key, a_ty, Some(expression_key))?;
        }

        self.tyc.impose(out_key.equate_with(expression_key))?;
        //dbg!(&out_key, &expression_key);

        Ok(out_key)
    }

    pub fn trigger_infer(&mut self, tr: &Trigger) -> Result<TcKey, TcErr<IAbstractType>> {
        let tr_key = *self.node_key.get_by_left(&NodeId::SRef(tr.sr)).expect("Added in constructor");
        let expression_key = self.expression_infer(&self.hir.expr(tr.sr), Some(IAbstractType::Bool))?;
        self.tyc.impose(tr_key.concretizes(expression_key))?;
        Ok(tr_key)
    }

    fn expression_infer(
        &mut self,
        exp: &Expression,
        target_type: Option<IAbstractType>,
    ) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        self.node_key.insert(NodeId::Expr(exp.eid), term_key);
        self.key_span.insert(term_key, exp.span.clone());
        if let Some(t) = target_type {
            self.tyc.impose(term_key.concretizes_explicit(t))?;
        }
        //dbg!(&exp.kind, term_key);
        match &exp.kind {
            ExpressionKind::LoadConstant(c) => {
                let cons_lit = match c {
                    Constant::BasicConstant(lit) => lit,
                    Constant::InlinedConstant(lit, anno_ty) => {
                        self.concretizes_annotated_type(term_key, anno_ty)?;
                        self.bind_to_annotated_type(term_key, anno_ty, None)?;
                        lit
                    }
                };
                let literal_type = self.match_const_literal(cons_lit);
                self.tyc.impose(term_key.concretizes_explicit(literal_type))?;
            }

            ExpressionKind::StreamAccess(sr, kind, args) => {
                if sr.is_input() {
                    assert!(args.is_empty(), "Parametrized Input Stream are unsupported");
                }

                if !args.is_empty() {
                    let target_stream: &Output =
                        self.hir.outputs().find(|o| o.sr == *sr).expect("Unable to find referenced stream");
                    let param_keys: Vec<_> = target_stream
                        .params
                        .iter()
                        .map(|p| {
                            let v = Variable { name: target_stream.name.clone() + "_" + &p.name };
                            self.tyc.get_var_key(&v)
                        })
                        .collect();
                    //dbg!(&param_keys);
                    let arg_keys: Vec<TcKey> = args
                        .iter()
                        .map(|arg| self.expression_infer(arg, None))
                        .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                    let res: Result<Vec<()>, TcErr<IAbstractType>> = param_keys
                        .iter()
                        .zip(arg_keys.iter())
                        .map(|(p, a)| self.tyc.impose(a.equate_with(*p)))
                        .collect();
                    res?;
                }

                let target_key = self.node_key.get_by_left(&NodeId::SRef(*sr)).expect("Entered in constructor");

                match kind {
                    StreamAccessKind::Sync => {
                        self.tyc.impose(term_key.equate_with(*target_key))?;
                    }
                    StreamAccessKind::DiscreteWindow(_wref) | StreamAccessKind::SlidingWindow(_wref) => {
                        //TODO use acutall wref as access methdd
                        let window = self.hir.single_window(Window { expr: exp.eid });
                        let (target_key, op, wait) = match window {
                            Either::Left(sw) => (self.node_key.get_by_left(&NodeId::SRef(sw.target)), sw.op, sw.wait),
                            Either::Right(dw) => (self.node_key.get_by_left(&NodeId::SRef(dw.target)), dw.op, dw.wait),
                        };
                        let target_key = *target_key.expect("Entered in Constructor");
                        //let duration_key = self.expression_infer(&*duration, None)?;
                        //self.tyc.impose(duration_key.concretizes_explicit(IAbstractType::Numeric))?;

                        use crate::ast::WindowOperation;
                        match op {
                            //Min|Max|Avg <T:Num> T -> Option<T>
                            WindowOperation::Min | WindowOperation::Max | WindowOperation::Average => {
                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                self.tyc.impose(inner_key.equate_with(target_key))?;
                            }
                            //Count: Any -> uint
                            WindowOperation::Count => {
                                self.tyc.impose(target_key.concretizes_explicit(IAbstractType::Any))?;
                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::UInteger(1)))?;
                            }
                            //integral :T <T:Num> -> T
                            //integral : T <T:Num> -> Float   <-- currently used
                            WindowOperation::Integral => {
                                self.tyc.impose(target_key.concretizes_explicit(IAbstractType::Numeric))?;
                                if wait {
                                    self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    //self.tyc.impose(inner_key.equate_with(ex_key))?;
                                    self.tyc.impose(inner_key.concretizes_explicit(IAbstractType::Float(1)))?;
                                } else {
                                    //self.tyc.impose(term_key.concretizes(ex_key))?;
                                    self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Float(1)))?;
                                }
                            }
                            //Σ and Π :T <T:Num> -> T
                            WindowOperation::Sum | WindowOperation::Product => {
                                self.tyc.impose(target_key.concretizes_explicit(IAbstractType::Numeric))?;
                                if wait {
                                    self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    self.tyc.impose(inner_key.equate_with(target_key))?;
                                } else {
                                    self.tyc.impose(term_key.concretizes(target_key))?;
                                }
                            }
                            //bool -> bool
                            WindowOperation::Conjunction | WindowOperation::Disjunction => {
                                self.tyc.impose(target_key.concretizes_explicit(IAbstractType::Bool))?;
                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                        }
                    }
                    StreamAccessKind::Hold => {
                        self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                        let inner_key = self.tyc.get_child_key(term_key, 0)?;
                        self.tyc.impose(target_key.equate_with(inner_key))?;
                    }
                    StreamAccessKind::Offset(off) => match off {
                        Offset::PastDiscreteOffset(_) | Offset::FutureDiscreteOffset(_) => {
                            self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                            self.tyc.impose(target_key.equate_with(inner_key))?;
                        }
                        Offset::FutureRealTimeOffset(d) | Offset::PastRealTimeOffset(d) => {
                            //dbg!("RealTimeOffset");
                            use num::rational::Rational64 as Rational;
                            use uom::si::frequency::hertz;
                            use uom::si::rational64::Frequency as UOM_Frequency;

                            use crate::tyc::pacing_types::AbstractPacingType::*;
                            //let n = UOM_Time::new::<second>(d);
                            let mut duration_as_f = d.as_secs_f64();
                            let mut c = 0;
                            while duration_as_f % 1.0f64 > 0f64 {
                                c += 1;
                                duration_as_f *= 10f64;
                            }
                            //dbg!(duration_as_f);
                            let rat = Rational::new(10i64.pow(c), duration_as_f as i64);
                            let freq = Freq::Fixed(UOM_Frequency::new::<hertz>(rat));
                            let target_ratio = self.pacing_tt[&NodeId::SRef(*sr)].expression_pacing.to_abstract_freq();
                            //TODO special case: period of current output > offset
                            // && offset is multiple of target stream (no optional needed)
                            if let Ok(Periodic(target_freq)) = target_ratio {
                                //fif the frequencies match the access is possible
                                //dbg!(&freq, &target_freq);
                                if let Ok(true) = target_freq.is_multiple_of(&freq) {
                                    //dbg!("frequencies compatible");
                                    self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    self.tyc.impose(target_key.equate_with(inner_key))?;
                                } else {
                                    //dbg!("frequencies NOT compatible");
                                    //if the ey dont match return error
                                    return Err(TcErr::Bound(
                                        *target_key,
                                        None,
                                        "Can't use realtime offset with non-matching frequencies".to_string(),
                                    ));
                                }
                            } else {
                                //Not a periodic target stream given
                                return Err(TcErr::Bound(
                                    *target_key,
                                    None,
                                    "Realtime offset on non periodic stream".to_string(),
                                ));
                            }
                        }
                    },
                };
            }
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr, None)?; //Option<X>
                let def_key = self.expression_infer(&*default, None)?; // Y
                                                                       //dbg!(ex_key, def_key);
                self.tyc.impose(ex_key.concretizes_explicit(IAbstractType::Option))?;
                let inner_key = self.tyc.get_child_key(ex_key, 0)?;
                //self.tyc.impose(def_key.equate_with(inner_key))?;
                //selftyc.impose(term_key.equate_with(def_key))?;
                self.tyc.impose(term_key.is_sym_meet_of(def_key, inner_key))?;
            }
            //TODO
            ///// implicit widening requieres join operand
            // a + b -> c c = meet(a,b) then equate a and b with join(a,b) //FIXME
            ExpressionKind::ArithLog(op, expr_v) => {
                use crate::hir::expression::ArithLogOp;
                let arg_keys: Result<Vec<TcKey>, TcErr<IAbstractType>> =
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
                                self.tyc.impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                                self.tyc.impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;

                                self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                                self.tyc.impose(term_key.equate_with(left_key))?;
                                self.tyc.impose(term_key.equate_with(right_key))?;
                            }
                            // Bool x Bool -> Bool
                            ArithLogOp::And | ArithLogOp::Or => {
                                self.tyc.impose(left_key.concretizes_explicit(IAbstractType::Bool))?;
                                self.tyc.impose(right_key.concretizes_explicit(IAbstractType::Bool))?;

                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                            // Any x Any -> Bool COMPARATORS
                            ArithLogOp::Eq
                            | ArithLogOp::Lt
                            | ArithLogOp::Le
                            | ArithLogOp::Ne
                            | ArithLogOp::Ge
                            | ArithLogOp::Gt => {
                                self.tyc.impose(left_key.equate_with(right_key))?;

                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                            ArithLogOp::Not | ArithLogOp::Neg | ArithLogOp::BitNot => {
                                unreachable!("unary operator cannot have 2 arguments")
                            }
                        }
                    }
                    1 => {
                        let arg_key = arg_keys[0];
                        match op {
                            // Bool -> Bool
                            ArithLogOp::Not => {
                                self.tyc.impose(arg_key.concretizes_explicit(IAbstractType::Bool))?;

                                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                            //Num -> Num
                            ArithLogOp::Neg | ArithLogOp::BitNot => {
                                self.tyc.impose(arg_key.concretizes_explicit(IAbstractType::Numeric))?;

                                self.tyc.impose(term_key.equate_with(arg_key))?;
                            }
                            _ => unreachable!("All other operators have 2 given arguments"),
                        }
                    }
                    _ => unreachable!(),
                }
            }
            ExpressionKind::Ite { condition, consequence, alternative } => {
                // Bool for condition - check given in the second argument
                self.expression_infer(&*condition, Some(IAbstractType::Bool))?;
                let cons_key = self.expression_infer(&*consequence, None)?; // X
                let alt_key = self.expression_infer(&*alternative, None)?; // X
                                                                           //Bool x T x T -> T
                self.tyc.impose(term_key.is_sym_meet_of(cons_key, alt_key))?;
            }

            ExpressionKind::Tuple(vec) => {
                let key_vec = vec
                    .iter()
                    .map(|ex| self.expression_infer(ex, None))
                    .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;
                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Tuple(vec.len())))?;
                for (n, child_key_inferred) in key_vec.iter().enumerate() {
                    let n_key_given = self.tyc.get_child_key(term_key, n)?;
                    self.tyc.impose(n_key_given.equate_with(*child_key_inferred))?;
                }
            }

            ExpressionKind::TupleAccess(expr, idx) => {
                let ex_key = self.expression_infer(expr, None)?;
                self.tyc.impose(ex_key.concretizes_explicit(IAbstractType::AnyTuple))?;

                let accessed_child = self.tyc.get_child_key(ex_key, *idx)?;
                self.tyc.impose(term_key.equate_with(accessed_child))?;
            }

            ExpressionKind::Widen(inner, ty) => {
                let inner_expr_key = self.expression_infer(inner, None)?;
                let (upper_bound, type_bound) = match ty {
                    AnnotatedType::UInt(x) => (IAbstractType::UInteger(1), IAbstractType::UInteger(*x)),
                    AnnotatedType::Int(x) => (IAbstractType::SInteger(1), IAbstractType::SInteger(*x)),
                    AnnotatedType::Float(x) => (IAbstractType::Float(1), IAbstractType::Float(*x)),
                    _ => unimplemented!("Unsupported widen Type"),
                };
                let internal_key = self.tyc.new_term_key();
                self.key_span.insert(internal_key, exp.span.clone());
                self.tyc.impose(internal_key.concretizes_explicit(type_bound))?;
                self.tyc.impose(inner_expr_key.concretizes_explicit(upper_bound))?;
                self.tyc.impose(internal_key.concretizes(inner_expr_key))?;
                self.tyc.impose(term_key.equate_with(internal_key))?;
            }
            ExpressionKind::Function { name, type_param, args } => {
                // check for name in context
                let fun_decl = self.hir.func_declaration(name);
                //Generics
                let generics: Vec<TcKey> = fun_decl
                    .generics
                    .iter()
                    .map(|gen| {
                        let gen_key: TcKey = self.tyc.new_term_key();
                        self.concretizes_annotated_type(gen_key, gen).map(|_| gen_key)
                    })
                    .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                for (t, gen) in type_param.iter().zip(generics.iter()) {
                    self.concretizes_annotated_type(*gen, t)?;
                    self.bind_to_annotated_type(*gen, t, None)?;
                }
                //FOR: type.captures(generic)
                args.iter()
                    .zip(fun_decl.parameters.iter())
                    .map(|(arg, param)| {
                        let p = self.replace_type(param, &generics)?;
                        let arg_key = self.expression_infer(&*arg, None)?;
                        self.tyc.impose(arg_key.concretizes(p))?;
                        Ok(arg_key)
                    })
                    .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                let return_type = self.replace_type(&fun_decl.return_type, &generics)?;

                self.tyc.impose(term_key.concretizes(return_type))?;
            }
            ExpressionKind::ParameterAccess(current_stream, ix) => {
                let output: &Output =
                    self.hir.outputs().find(|o| o.sr == *current_stream).expect("Expect valid stream reference");
                let v = Variable { name: output.name.clone() + "_" + &output.params[*ix].name };
                let par_key = self.tyc.get_var_key(&v);
                //dbg!(par_key);
                self.tyc.impose(term_key.equate_with(par_key))?;
            }
        };

        Ok(term_key)
    }

    fn replace_type(&mut self, at: &AnnotatedType, to: &[TcKey]) -> Result<TcKey, TcErr<IAbstractType>> {
        match at {
            AnnotatedType::Param(idx, _) => Ok(to[*idx]),
            AnnotatedType::Numeric
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
            }
        }
    }

    pub(crate) fn check_explicit_bounds(
        annotated_checks: HashMap<TcKey, (IConcreteType, Option<TcKey>)>,
        tt: &TypeTable<IAbstractType>,
    ) -> Vec<TcErr<IAbstractType>> {
        annotated_checks
            .into_iter()
            .filter_map(|(target, (bound, conflict))| {
                let resolved = tt[&target].clone();
                if resolved != bound {
                    Some(TcErr::Bound(target, conflict, "Inferred type does not match annotated type".to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub(crate) fn handle_error(&self, err: TcErr<IAbstractType>) -> String {
        let primal_key;
        let msg = match err {
            TcErr::ChildAccessOutOfBound(key, ty, n) => {
                primal_key = key;
                format!(
                    "Invalid child-type access by {:?}: {}-th child for {:?}",
                    self.node_key.get_by_right(&key).unwrap(),
                    n,
                    ty
                )
            }
            TcErr::KeyEquation(k1, k2, msg) => {
                primal_key = k2;
                format!(
                    "Incompatible Type for equation of {:?} and {:?}: {}",
                    self.node_key.get_by_right(&k1),
                    self.node_key.get_by_right(&k2),
                    msg
                )
            }
            TcErr::Bound(key, key2, msg) => {
                primal_key = key;
                match key2 {
                    None => format!("Invalid type bound enforced on {:?}: {}", self.node_key.get_by_right(&key), msg),
                    Some(k2) => format!(
                        "Invalid type bound enforced on {:?} by {:?}: {}",
                        self.node_key.get_by_right(&key).unwrap(),
                        self.node_key.get_by_right(&k2),
                        msg
                    ),
                }
            }
            TcErr::ArityMismatch { key, variant, inferred_arity, reported_arity } => {
                primal_key = key;
                format!(
                    "Arity Clash: Found type {:?} for key {:?} inferred arity: {}  but got: {}",
                    variant, key, inferred_arity, reported_arity
                )
            }
            TcErr::Construction(key, bound1, err) => {
                primal_key = key;
                format!("Could not construct concrete type for: {:?} of key: {:?}\nreason: {}", bound1, key, err)
            }
            TcErr::ChildConstruction(parent_key, index, child, reason) => {
                primal_key = parent_key;
                format!(
                    "Could not construct concrete type for: {:?} of parent: {:?} at index {}\nreason: {}",
                    child, parent_key, index, reason
                )
            }
        };
        if let Some(error_key_span) = self.key_span.get(&primal_key) {
            self.handler.error_with_span("Stream inference error", error_key_span.clone(), Some(&msg));
        } else {
            self.handler.error(&msg);
        }
        msg
    }
}

/*
fn get_abstract_type_of_string_value(value_str: &str) -> Result<IAbstractType, String> {
    let int_parse = value_str.parse::<i64>();
    let uint_parse = value_str.parse::<u64>();
    match (&int_parse, &uint_parse) {
        //(Ok(s),Ok(u)) => return Ok(IAbstractType::SInteger(64 - s.leading_zeros())),
        (Ok(_s), Ok(_u)) => return Ok(IAbstractType::Integer),
        (Err(_), Ok(u)) => return Ok(IAbstractType::UInteger(64 - u.leading_zeros())),
        (Ok(s), Err(_)) => {
            let n = if s.is_negative() { 1 } else { 0 } + s.abs().leading_zeros();
            return Ok(IAbstractType::SInteger(64 - n));
        }
        (Err(_), Err(_)) => {}
    }

    let float_parse = value_str.parse::<f64>();
    if float_parse.is_ok() {
        return Ok(IAbstractType::Float(32));
    }

    if &value_str[0..1] == "\"" && &value_str[value_str.len() - 1..value_str.len()] == "\"" {
        return Ok(IAbstractType::TString);
    }
    Err(format!("Non matching String Literal: {}", value_str))
}
*/

#[cfg(test)]
mod value_type_tests {
    use crate::common_ir::StreamReference;
    use crate::hir::modes::IrExpression;
    use crate::reporting::Handler;
    use crate::tyc::rtltc::NodeId;
    use crate::tyc::value_types::IConcreteType;
    use crate::tyc::LolaTypeChecker;
    use crate::RTLolaAst;
    use crate::RTLolaHIR;
    use std::collections::HashMap;
    use std::path::PathBuf;

    struct TestBox {
        pub hir: RTLolaHIR<IrExpression>,
        pub handler: Handler,
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
        let handler = crate::reporting::Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst = match crate::parse::parse(spec, &handler, crate::FrontendConfig::default()) {
            Ok(s) => s,
            Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
        };
        let hir = crate::hir::RTLolaHIR::<IrExpression>::transform_expressions(
            ast,
            &handler,
            &crate::FrontendConfig::default(),
        );
        //let mut dec = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        TestBox { hir, handler }
    }

    fn complete_check(spec: &str) -> usize {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pacing_tt = ltc.pacing_type_infer().unwrap();
        assert!(ltc.value_type_infer(&pacing_tt).is_ok());
        test_box.handler.emitted_errors()
    }

    fn check_value_type(spec: &str) -> (TestBox, HashMap<NodeId, IConcreteType>) {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pacing_tt = ltc.pacing_type_infer().expect("Expected valid pacing type");
        let tt_result = ltc.value_type_infer(&pacing_tt);
        if let Err(ref e) = tt_result {
            eprintln!("{}", e.clone());
        }
        assert!(tt_result.is_ok(), "Expect Valid Input - Value Type check failed");
        let tt = tt_result.expect("ensured by assertion");
        (test_box, tt)
    }

    fn check_expect_error(spec: &str) -> TestBox {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pt = ltc.pacing_type_infer().expect("expect valid pacing input");
        let tt_result = ltc.value_type_infer(&pt);
        assert!(tt_result.is_err(), "Expected error in value type result");
        println!("{}", tt_result.err().unwrap());
        test_box
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn direct_implication() {
        let spec = "input i: Int8\noutput o := i";
        let (tb, result_map) = check_value_type(spec);
        let input_sr = tb.hir.get_input_with_name("i").unwrap().sr;
        let output_sr = tb.hir.get_output_with_name("o").unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_sr)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_sr)], IConcreteType::Integer8);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn direct_widening() {
        let spec = "input i: Int8\noutput o :Int32 := widen<Int32>(i)";
        let (tb, result_map) = check_value_type(spec);
        let input_sr = tb.hir.get_input_with_name("i").unwrap().sr;
        let output_sr = tb.hir.get_output_with_name("o").unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_sr)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_sr)], IConcreteType::Integer32);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn integer_addition_wideing() {
        let spec = "input i: Int8\ninput i1: Int16\noutput o := widen<Int16>(i) + i1";
        let (tb, result_map) = check_value_type(spec);
        let mut input_iter = tb.hir.inputs();
        let input_i_id = input_iter.next().unwrap().sr;
        let input_i1_id = input_iter.next().unwrap().sr;
        let output_id = tb.hir.outputs().next().unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_i_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(input_i1_id)], IConcreteType::Integer16);
        assert_eq!(result_map[&NodeId::SRef(output_id)], IConcreteType::Integer16);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn parametric_access_default() {
        let spec = "output i(a: Int8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true):= if b then a else 0 \n output o(x) spawn @1Hz with 42 := i(1,false).offset(by:-1).defaults(to: 42)";
        let (tb, result_map) = check_value_type(spec);
        let o2_sr = tb.output("i");
        let o1_id = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(o1_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(o2_sr)], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_x() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true) := 1";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(output_sr)], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_infer() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz spawn @1Hz with (5, true) := a";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(0, complete_check(spec));
        assert_eq!(0, tb.handler.emitted_errors());
        assert_eq!(result_map[&NodeId::SRef(output_sr)], IConcreteType::UInteger8);
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz spawn @1Hz with (5, true) := 1 output y @1Hz := x(1, false).hold().defaults(to:5)";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.output("x");
        let output_2_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(output_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(output_2_id)], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_two_many() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz spawn @1Hz with (5, true, false) := a";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn parametric_declaration_param_two_few() {
        let spec = "output x(a: UInt8, b: Bool, c:String) @1Hz spawn @1Hz with (5, true) := a";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn parametric_declaration_param_infer2() {
        let spec = "output x (a, b) @1Hz spawn @1Hz with (5, true) := 42";
        let (tb, result_map) = check_value_type(spec);
        let x = tb.output("x");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::Param(0, x)], IConcreteType::Integer32);
        assert_eq!(result_map[&NodeId::Param(1, x)], IConcreteType::Bool);
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(sr)], IConcreteType::Float32);
    }

    #[test]
    //TODO float16 annotation is treaded as Floa32, as Float16 is not implemented as Concrete Type
    fn simple_const_float16() {
        let spec = "constant c: Float16 := 2.1 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(sr)], IConcreteType::Float32);
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3 output o @1Hz := c";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(sr)], IConcreteType::Integer8);
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true output o @1Hz := c";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2 output o @1Hz := c";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3 output o @1Hz := c";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_valid_coersion() {
        //TODO does not check output type, only validity
        for spec in &[
            "constant c: Int8 := 1\noutput o: Int32 @1Hz:= widen<Int32>(c)",
            "constant c: UInt16 := 1\noutput o: UInt64 @1Hz:= widen<UInt64>(c)",
            "constant c: Float32 := 1.0\noutput o: Float64 @1Hz:= widen<Float64>(c)",
        ] {
            assert_eq!(0, complete_check(spec));
        }
    }

    #[test]
    fn simple_invalid_conversion() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 @1Hz := c";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_invalid_wideing() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 @1Hz := widen<Int8>(c)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_explicit_widening() {
        let spec = "constant c: Int32 := 1\n constant d: Int8 := 2\noutput o @1Hz := c + widen<Int32>(d)";
        let (tb, result_map) = check_value_type(spec);
        let sr = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(sr)], IConcreteType::Integer32);
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger false";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.hir.triggers().nth(0).unwrap().sr;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(tr_id)], IConcreteType::Bool);
    }

    #[test]
    fn simple_trigger_message() {
        let spec = "trigger false \"alert always\"";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.hir.triggers().nth(0).unwrap().sr;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(tr_id)], IConcreteType::Bool);
    }

    #[test]
    fn faulty_trigger() {
        let spec = "trigger 1";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 @1Hz := 3 + 5";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.hir.outputs().next().unwrap().sr;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
    }

    #[test]
    fn simple_binary_input() {
        let spec = "input i: Int8\noutput o: Int8 := 3 + i";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let in_id = tb.input("i");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
    }

    #[test]
    fn simple_unary() {
        let spec = "output o @1Hz:= !false \n\
                               output u: Bool @1Hz:= !false";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let out_id_2 = tb.output("u");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
        assert_eq!(result_map[&NodeId::SRef(out_id_2)], IConcreteType::Bool);
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool @1Hz:= !3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
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
            assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
        }

        let in_id = tb.input("i_0");
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Bool);
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 @1Hz := false + 2.5";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 @1Hz := if false then 1 else 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
    }

    #[test]
    fn simple_ite_compare() {
        let spec = "output e :Int8 @1Hz := if 1 == 0 then 0 else -1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("e");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o @1Hz := if !false then 1.3 else -2.0";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 @1Hz := if 3 then 1 else 1";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 @1Hz := if true then 1 else false";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_parenthesized_expr() {
        let spec = "input s: String\noutput o: Bool := s[-1].defaults(to: \"\") == \"a\"";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        let in_id = tb.input("s");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::TString);
    }

    #[test]
    fn test_underspecified_type() {
        //Default for num literals applied
        let spec = "output o @1Hz := 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("o");
        assert_eq!(0, complete_check(spec));
        let res_type = &result_map[&NodeId::SRef(out_id)];
        assert!(*res_type == IConcreteType::Integer32 || *res_type == IConcreteType::Integer64);
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        let in_id = tb.input("a");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 @1Hz:= 3\n output b: UInt8 := a[0]";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 @1Hz := 3\n output b: UInt8 := a[-1].defaults(to: 3)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("b");
        let in_id = tb.output("a");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_offset_regression() {
        let spec = "input a: UInt8 \n output sum := sum[-1].defaults(to: 0) + a";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("a");
        let out_id = tb.output("sum");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 @1Hz:= 3\n output b: Bool := a[-1].defaults(to: false)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }
    #[test]
    //#[ignore] // paramertic streams need new design after syntax revision
    fn test_filter_type() {
        let spec = "input in: Bool\n output a: Int8 filter in := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Bool);
    }

    #[test]
    //#[ignore] // paramertic streams need new design after syntax revision
    fn test_filter_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 filter in := 3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_close_type() {
        let spec = "input in: Bool\n output a(b: Bool): Int8 @1Hz close in := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Bool);
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_close_type_faulty() {
        //Close condition non boolean type
        let spec = "input in: Int8\n output a(b: Bool): Int8 @1Hz close in := 3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_param_spec() {
        let spec = "output a(p1: Int8): Int8 @1Hz:= 3 output b: Int8 := a(3)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_param_spec_faulty() {
        let spec = "output a(p1: Int8): Int8 @1Hz:= 3 output b: Int8 := a(true)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_param_inferred() {
        let spec = "input i: Int8 output x(param): Int8 := i output y: Int8 := x(i)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("x");
        let out_id2 = tb.output("y");
        let in_id = tb.input("i");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_param_inferred_conflicting() {
        let spec = "input i: Int8, j: UInt8 output x(param): Int8 := i output y: Int8 := x(i) output z: Int8 := x(j)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] //Todo: Fix when tuples are implemented
    fn test_lookup_incomp() {
        let spec = "output a(p1: Int8): Int8 @1Hz:= 3\n output b: UInt8 := a(3)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) @1Hz:= (14, false)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&NodeId::SRef(out_id)],
            IConcreteType::Tuple(vec![IConcreteType::Integer8, IConcreteType::Bool])
        );
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) @1Hz := (14, 3)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_tuple_access() {
        //TODO runs with 'in.1' not with 'in[0].1' - zero offset still optional result
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&NodeId::SRef(in_id)],
            IConcreteType::Tuple(vec![IConcreteType::Integer8, IConcreteType::Bool])
        );
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
    }

    #[test]
    fn test_tuple_access_faulty_type() {
        //TODO optional result at offset zero
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].0";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_tuple_access_faulty_len() {
        //TODO optional result at offset zero
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.2";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_optional_type() {
        let spec = "input in: Int8\noutput out: Int8? := in.offset(by: -1)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Option(IConcreteType::Integer8.into()));
    }

    #[test]
    fn test_optional_type_faulty() {
        let spec = "input in: Int8\noutput out: Int8? := in";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("a");
        let out_id = tb.output("b");
        //assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_tuple_of_tuples() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Int16 := widen<Int16>(in[0].0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        let input_type = IConcreteType::Tuple(vec![
            IConcreteType::Integer8,
            IConcreteType::Tuple(vec![IConcreteType::UInteger8, IConcreteType::Bool]),
        ]);
        assert_eq!(result_map[&NodeId::SRef(in_id)], input_type);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer16);
    }

    #[test]
    #[ignore] //tuple with variable length are not supported TODO
    fn test_tuple_of_tuples2() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Bool := in.1.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        let input_type = IConcreteType::Tuple(vec![
            IConcreteType::Integer8,
            IConcreteType::Tuple(vec![IConcreteType::UInteger8, IConcreteType::Bool]),
        ]);
        assert_eq!(result_map[&NodeId::SRef(in_id)], input_type);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
    }

    #[test]
    fn test_tuple_of_tuples3() {
        let spec = "output out: Bool @1Hz := (5, (7.0, true)).1.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Bool);
    }

    #[test]
    fn test_tuple_of_tuples4() {
        let spec = "output out1 := out2.1.1\noutput out2 @1Hz := (5, (7.0, true))";
        let (tb, result_map) = check_value_type(spec);
        let out2 = tb.output("out2");
        let out1 = tb.output("out1");
        let tuple_type = IConcreteType::Tuple(vec![
            IConcreteType::Integer32,
            IConcreteType::Tuple(vec![IConcreteType::Float32, IConcreteType::Bool]),
        ]);
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out2)], tuple_type);
        assert_eq!(result_map[&NodeId::SRef(out1)], IConcreteType::Bool);
    }

    #[test]
    fn test_faulty_option_access() {
        let spec = "input x:Int32\noutput out @1Hz := x.hold().0";
        check_expect_error(spec);
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 @5Hz:= in.aggregate(over: 3s, using: Σ)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer64);
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
    }

    /* invalid for pacing type -> make pacing type check
    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in.aggregate(over: 3s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }
    */

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 3s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    /*
    //currently rejected in new hir
    #[test]
    fn test_window_invalid_duration() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 0s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: -3s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }
    */

    #[test]
    #[ignore] // uint to int cast currently not allowed
    fn test_aggregation_implicit_cast() {
        let spec = "input in: UInt8\n\
             output out: Int16 @5Hz := in.aggregate(over_exactly: 3s, using: Σ).defaults(to: 5)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer16);
    }

    #[test]
    #[ignore] //int to float cast currently not allowed
    fn test_aggregation_implicit_cast2() {
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: avg).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
    }

    #[test]
    //#[ignore] //symmetric type relation extends input int8 to float
    fn test_aggregation_implicit_cast3() {
        let spec =
                            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
    }

    #[test]
    fn test_aggregation_integer_integral() {
        let spec =
                            "input in: UInt8\n output out: UInt8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
        let spec =
            "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
        let spec =
            "input in: UInt8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
        let spec =
            "input in: Int8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("in");
        let out_id = tb.output("out");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := widen<Float64>(velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0))";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("velo");
        let out_id = tb.output("avg");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Float32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float64);
    }

    #[test]
    fn test_rt_offset() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to:1)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    fn test_rt_offset_regression() {
        let spec = "output a @10Hz := a.offset(by: -100ms).defaults(to:0) + 1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer32);
    }

    #[test]
    fn test_rt_offset_regression2() {
        let spec = "
                            output x @ 10Hz := 1
                            output x_diff := x - x.offset(by:-1s).defaults(to: x)
                        ";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("x");
        let out_id2 = tb.output("x_diff");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer32);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer32);
    }

    #[test]
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-1s].defaults(to:1)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    fn test_rt_offset_skip2() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-2s].defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger8);
    }

    #[test]
    #[ignore] //implicit casting not usable currently
    fn test_casting_implicit_types() {
        let spec = "input x: UInt8\noutput y: Float32 := cast(x)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::UInteger8);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Float32);
    }

    #[test]
    fn test_casting_explicit_types() {
        let spec = "input x: Int32\noutput y: UInt32 := cast<Int32,UInt32>(x)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.input("x");
        let out_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Integer32);
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::UInteger32);
    }

    #[test]
    fn infinite_recursion_regression() {
        // this should fail in type checking as the value type of `c` cannot be determined.
        // it currently fails because default expects a optional value
        let spec = "output c @1Hz := c.defaults(to:0)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_simple_annotated() {
        let spec = "output c: Int8 @1Hz := 42";
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn test_simple_annotated2() {
        let spec = "output c: Int8 @1Hz := widen<Int32>(42)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    /* //TODO: I don't know what this test wants me to check as the table getters are magical.
    #[test]
    fn test_function_arguments_regression() {
        let spec = "input a: Int32\ntrigger a > 50";
        let type_table = type_check(spec);
        // expression `a > 50` has NodeId = 3
        let exp_a_gt_50_id = NodeId::new(5);
        assert_eq!(type_table.get_value_type(exp_a_gt_50_id), &ValueTy::Bool);
        assert_eq!(type_table.get_func_arg_types(exp_a_gt_50_id), &vec![ValueTy::Int(IntTy::I32)]);
    }
    */

    /* Tests that now fail already in pacing - not needed here, but are here so they may not be forgotten
    #[test]
    #[ignore] //Pacing analysis fails already no need to test again
    fn test_rt_offset_fail() {
        let spec = "output a: Int8 @0.5Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] //Unnecessary hold() is now an Error in pacing analysis
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x.hold().defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.hir.inputs().next().unwrap().sr;
        let out_id = tb.hir.outputs().next().unwrap().sr;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&out_id], IConcreteType::UInteger8);
    }
    */
}
