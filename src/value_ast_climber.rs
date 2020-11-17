use super::*;
extern crate regex;

use crate::pacing_types::{ConcretePacingType, Freq};
use crate::rtltc::NodeId;
use crate::value_types::IAbstractType;
use bimap::BiMap;
use front::common_ir::Offset;
use front::hir::expression::{
    Constant, ConstantLiteral, Expression, ExpressionKind, StreamAccessKind,
};
use front::hir::modes::ir_expr::WithIrExpr;
use front::hir::modes::HirMode;
use front::hir::{AnnotatedType, Input, Output, Trigger, Window};
use front::reporting::{Handler, Span};
use front::RTLolaHIR;
use itertools::Either;
use rusttyc::types::Abstract;
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
    pub(crate) pacing_tt: HashMap<NodeId, ConcretePacingType>,
}

impl<'a, M> ValueContext<'a, M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub fn new(
        hir: &'a RTLolaHIR<M>,
        //decl: DeclarationTable,
        handler: &'a Handler,
        pacing_tt: HashMap<NodeId, ConcretePacingType>,
    ) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = BiMap::new();
        let key_span = HashMap::new();

        for input in hir.inputs() {
            let key = tyc.get_var_key(&Variable {
                name: input.name.clone(),
            });
            node_key.insert(NodeId::SRef(input.sr), key);
            //key_span.insert(key, input.span);
        }

        for out in hir.outputs() {
            let key = tyc.get_var_key(&Variable {
                name: out.name.clone(),
            });
            dbg!(key);
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
        }
    }

    pub fn input_infer(&mut self, input: &Input) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = *self
            .node_key
            .get_by_left(&NodeId::SRef(input.sr))
            .expect("Added in constructor");
        //Annotated Type

        let annotated_type_replaced = self.match_annotated_type(&input.annotated_type);
        //can skip any case as type must be provided
        self.tyc
            .impose(term_key.has_exactly_type(annotated_type_replaced))?;

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
        let out_key = *self
            .node_key
            .get_by_left(&NodeId::SRef(out.sr))
            .expect("Added in constructor");

        let annotated_type_replaced = out
            .annotated_type
            .as_ref()
            .map(|ty| self.match_annotated_type(ty))
            .unwrap_or(IAbstractType::Any);
        dbg!(&annotated_type_replaced);
        if let IAbstractType::Any = annotated_type_replaced {
        } else {
            //self.tyc.impose(out_key.concretizes_explicit(annotated_type_replaced.clone()))?;
            self.tyc
                .impose(out_key.has_exactly_type(annotated_type_replaced))?;
        }

        //let mut param_types = Vec::new();
        for param in &out.params {
            let param_key = self.tyc.get_var_key(&Variable {
                name: out.name.clone() + "_" + &param.name.clone(),
            });
            dbg!(param_key);
            self.node_key
                .insert(NodeId::Param(param.idx, out.sr), param_key);
            //self.key_span.insert(param_key, param.span);

            let t = param
                .annotated_type
                .as_ref()
                .map(|t| self.match_annotated_type(t))
                .unwrap_or(IAbstractType::Any);
            self.tyc.impose(param_key.concretizes_explicit(t))?;
            //param_types.push(param_key);
        }

        dbg!(&out.instance_template);
        let opt_spawn = &self.hir.spawn(out.sr);
        if let Some((spawn, opt_cond)) = opt_spawn {
            //chek target exression type matches parameter type
            let _target_expr_key = self.expression_infer(spawn, Some(IAbstractType::Bool))?;
            let _cond_key = opt_cond.map(|e| self.expression_infer(e, Some(IAbstractType::Bool)));
        }
        if let Some(close) = &self.hir.close(out.sr) {
            self.expression_infer(close, Some(IAbstractType::Bool))?;
        }
        if let Some(filter) = &self.hir.filter(out.sr) {
            self.expression_infer(filter, Some(IAbstractType::Bool))?;
        }

        let expression_key = self.expression_infer(self.hir.expr(out.sr), None)?;
        dbg!(&out_key, &expression_key);
        self.tyc.impose(out_key.equate_with(expression_key))?;
        Ok(out_key)
    }

    pub fn trigger_infer(&mut self, tr: &Trigger) -> Result<TcKey, TcErr<IAbstractType>> {
        let tr_key = *self
            .node_key
            .get_by_left(&NodeId::SRef(tr.sr))
            .expect("Added in constructor");
        let expression_key =
            self.expression_infer(&self.hir.expr(tr.sr), Some(IAbstractType::Bool))?;
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
        dbg!(&exp.kind, term_key);
        match &exp.kind {
            ExpressionKind::LoadConstant(c) => {
                let (cons_lit, anno_ty) = match c {
                    Constant::BasicConstant(lit) => (lit, IAbstractType::Any),
                    Constant::InlinedConstant(lit, anno_ty) => {
                        (lit, self.match_annotated_type(anno_ty))
                    }
                };
                let literal_type = self.match_const_literal(cons_lit);
                dbg!(&literal_type);
                self.tyc
                    .impose(term_key.concretizes_explicit(literal_type))?;
                if !matches!(anno_ty, IAbstractType::Any) {
                    self.tyc.impose(term_key.has_exactly_type(anno_ty))?;
                }
            }

            ExpressionKind::StreamAccess(sr, kind, args) => {
                //let ex_key = self.expression_infer(&self.hir.expr(sr), None)?;
                /*
                let target_key = match sr {
                    StreamReference::OutRef(ix) => self.hir.outputs().nth(ix).expect("Idx of SRef is always valid"),
                    StreamReference::InRef(ix) => self.hir.inputs().nth(ix).expect("Idx of SRef is always valid"),
                }
                let target_stream: &Output = self
                    .hir
                    .outputs()
                    .nth(sr.out_ix())
                    .expect("Idx of SRef is always valid");
                */

                if sr.is_input() {
                    assert!(args.is_empty(), "Parametrized Input Stream are unsupported");
                }

                if !args.is_empty() {
                    let target_stream: &Output = self
                        .hir
                        .outputs()
                        .nth(sr.out_ix())
                        .expect("Idx of SRef is always valid");
                    let param_keys: Vec<_> = target_stream
                        .params
                        .iter()
                        .map(|p| {
                            let v = Variable {
                                name: target_stream.name.clone() + "_" + &p.name,
                            };
                            self.tyc.get_var_key(&v)
                        })
                        .collect();
                    let arg_keys: Result<Vec<TcKey>, TcErr<IAbstractType>> = args
                        .iter()
                        .map(|arg| self.expression_infer(arg, None))
                        .collect();
                    let arg_keys = arg_keys?;

                    let res: Result<Vec<()>, TcErr<IAbstractType>> = param_keys
                        .iter()
                        .zip(arg_keys.iter())
                        .map(|(p, a)| self.tyc.impose(a.concretizes(*p)))
                        .collect();
                    res?;
                }

                let target_key = self
                    .node_key
                    .get_by_left(&NodeId::SRef(*sr))
                    .expect("Entered in constructor");

                match kind {
                    StreamAccessKind::Sync => {
                        self.tyc.impose(term_key.equate_with(*target_key))?;
                    }
                    StreamAccessKind::DiscreteWindow(_wref)
                    | StreamAccessKind::SlidingWindow(_wref) => {
                        //TODO use acutall wref as access methdd
                        let window = self.hir.single_window(Window { expr: exp.eid });
                        let (target_key, op, wait) = match window {
                            Either::Left(sw) => (
                                self.node_key.get_by_left(&NodeId::SRef(sw.target)),
                                sw.op,
                                sw.wait,
                            ),
                            Either::Right(dw) => (
                                self.node_key.get_by_left(&NodeId::SRef(dw.target)),
                                dw.op,
                                dw.wait,
                            ),
                        };
                        let target_key = *target_key.expect("Entered in Constructor");
                        //let duration_key = self.expression_infer(&*duration, None)?;
                        //self.tyc.impose(duration_key.concretizes_explicit(IAbstractType::Numeric))?;

                        use front::ast::WindowOperation;
                        match op {
                            //Min|Max|Avg <T:Num> T -> Option<T>
                            WindowOperation::Min
                            | WindowOperation::Max
                            | WindowOperation::Average => {
                                self.tyc.impose(term_key.concretizes_explicit(
                                    IAbstractType::Option(IAbstractType::Any.into()),
                                ))?;
                                let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                self.tyc.impose(inner_key.equate_with(target_key))?;
                            }
                            //Count: Any -> uint
                            WindowOperation::Count => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(IAbstractType::Any))?;
                                self.tyc.impose(
                                    term_key.concretizes_explicit(IAbstractType::UInteger(1)),
                                )?;
                            }
                            //integral :T <T:Num> -> T
                            //integral : T <T:Num> -> Float   <-- currently used
                            WindowOperation::Integral => {
                                self.tyc.impose(
                                    target_key.concretizes_explicit(IAbstractType::Numeric),
                                )?; //TODO maybe numeric
                                if wait {
                                    self.tyc.impose(term_key.concretizes_explicit(
                                        IAbstractType::Option(IAbstractType::Any.into()),
                                    ))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    //self.tyc.impose(inner_key.equate_with(ex_key))?;
                                    self.tyc.impose(
                                        inner_key.concretizes_explicit(IAbstractType::Float(1)),
                                    )?;
                                } else {
                                    //self.tyc.impose(term_key.concretizes(ex_key))?;
                                    self.tyc.impose(
                                        term_key.concretizes_explicit(IAbstractType::Float(1)),
                                    )?;
                                }
                            }
                            //Σ and Π :T <T:Num> -> T
                            WindowOperation::Sum | WindowOperation::Product => {
                                self.tyc.impose(
                                    target_key.concretizes_explicit(IAbstractType::Numeric),
                                )?;
                                if wait {
                                    self.tyc.impose(term_key.concretizes_explicit(
                                        IAbstractType::Option(IAbstractType::Any.into()),
                                    ))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    self.tyc.impose(inner_key.equate_with(target_key))?;
                                } else {
                                    self.tyc.impose(term_key.concretizes(target_key))?;
                                }
                            }
                            //bool -> bool
                            WindowOperation::Conjunction | WindowOperation::Disjunction => {
                                self.tyc
                                    .impose(target_key.concretizes_explicit(IAbstractType::Bool))?;
                                self.tyc
                                    .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                        }
                    }
                    StreamAccessKind::Hold => {
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Option(
                                IAbstractType::Any.into(),
                            )))?;
                        let inner_key = self.tyc.get_child_key(term_key, 0)?;
                        self.tyc.impose(target_key.equate_with(inner_key))?;
                    }
                    StreamAccessKind::Offset(off) => match off {
                        Offset::PastDiscreteOffset(_) | Offset::FutureDiscreteOffset(_) => {
                            self.tyc.impose(term_key.concretizes_explicit(
                                IAbstractType::Option(IAbstractType::Any.into()),
                            ))?;
                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                            self.tyc.impose(target_key.equate_with(inner_key))?;
                        }
                        Offset::FutureRealTimeOffset(d) | Offset::PastRealTimeOffset(d) => {
                            use num::rational::Rational64 as Rational;
                            use uom::si::frequency::hertz;
                            use uom::si::rational64::Frequency as UOM_Frequency;

                            use crate::pacing_types::AbstractPacingType::*;
                            //let n = UOM_Time::new::<second>(d);
                            let mut duration_as_f = d.as_secs_f64();
                            let mut c = 0;
                            while duration_as_f % 1.0f64 > 0f64 {
                                c += 1;
                                duration_as_f *= 10f64;
                            }
                            let rat = Rational::new(10i64.pow(c), duration_as_f as i64);
                            let freq = Freq::Fixed(UOM_Frequency::new::<hertz>(rat));
                            let target_ratio =
                                self.pacing_tt[&NodeId::SRef(*sr)].to_abstract_freq();
                            if let Ok(Periodic(target_freq)) = target_ratio {
                                //fif the frequencies match no optional needed
                                if let Ok(true) = freq.is_multiple_of(&target_freq) {
                                    self.tyc.impose(term_key.equate_with(*target_key))?;
                                } else {
                                    //if the ey dont match return optional
                                    self.tyc.impose(term_key.concretizes_explicit(
                                        IAbstractType::Option(IAbstractType::Any.into()),
                                    ))?;
                                    let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                    self.tyc.impose(target_key.equate_with(inner_key))?;
                                }
                            } else {
                                //Not a periodic target stream given
                                return Err(TcErr::Bound(
                                    *target_key,
                                    None,
                                    "Realtime offset on non periodic stream".to_string(),
                                ));
                            }
                            /*
                            front::ast::Offset::RealTime(r, unit) => {
                                //if periode < offset -> optinal
                                use num::rational::Rational64 as Rational;
                                use uom::si::frequency::hertz;
                                use uom::si::rational64::Frequency as UOM_Frequency;
                                use uom::si::rational64::Time as UOM_Time;
                                use uom::si::time::second;

                                if *r.numer() == 0 {
                                    self.tyc.impose(term_key.equate_with(ex_key))?;
                                } else if *r.numer() > 0 {
                                    return Err(TcErr::Bound(
                                        term_key,
                                        None,
                                        "Found positive realtime offset - not yet supported".to_string(),
                                    ));
                                } else {
                                    use crate::pacing_types::AbstractPacingType::*;
                                    let uom_offset_duration = offset.to_uom_time().unwrap();
                                    let freq = Freq::Fixed(UOM_Frequency::new::<hertz>(
                                        Rational::from_integer(1) / uom_offset_duration.get::<second>(),
                                    ));
                                    let target_ratio = self.pacing_tt[&target_expr.id].to_abstract_freq();
                                    if let Ok(Periodic(target_freq)) = target_ratio {
                                        //fif the frequencies match no optional needed
                                        if let Ok(true) = freq.is_multiple_of(&target_freq) {
                                            self.tyc.impose(term_key.equate_with(ex_key))?;
                                        } else {
                                            //if the ey dont match return optional
                                            self.tyc.impose(term_key.concretizes_explicit(
                                                IAbstractType::Option(IAbstractType::Any.into()),
                                            ))?;
                                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                                            self.tyc.impose(ex_key.equate_with(inner_key))?;
                                        }
                                    } else {
                                        //Not a periodic target stream given
                                        return Err(TcErr::Bound(
                                            ex_key,
                                            None,
                                            "Realtime offset on non periodic stream".to_string(),
                                        ));
                                    }
                                }
                                //unimplemented!("RealTime offset not yet supported in Value Type inference")
                            }
                            */
                        }
                    },
                };
            }
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr, None)?; //Option<X>
                let def_key = self.expression_infer(&*default, None)?; // Y
                dbg!(ex_key, def_key);
                self.tyc.impose(
                    ex_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())),
                )?;
                let inner_key = self.tyc.get_child_key(ex_key, 0)?;
                //self.tyc.impose(def_key.equate_with(inner_key))?;
                //selftyc.impose(term_key.equate_with(def_key))?;
                self.tyc
                    .impose(term_key.is_sym_meet_of(def_key, inner_key))?;
            }
            //TODO
            ///// implicit widening requieres join operand
            // a + b -> c c = meet(a,b) then equate a and b with join(a,b) //FIXME
            ExpressionKind::ArithLog(op, expr_v) => {
                use front::hir::expression::ArithLogOp;
                let arg_keys: Result<Vec<TcKey>, TcErr<IAbstractType>> = expr_v
                    .iter()
                    .map(|expr| self.expression_infer(expr, None))
                    .collect();
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
                                self.tyc.impose(
                                    left_key.concretizes_explicit(IAbstractType::Numeric),
                                )?;
                                self.tyc.impose(
                                    right_key.concretizes_explicit(IAbstractType::Numeric),
                                )?;

                                self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                                self.tyc.impose(term_key.equate_with(left_key))?;
                                self.tyc.impose(term_key.equate_with(right_key))?;
                            }
                            // Bool x Bool -> Bool
                            ArithLogOp::And | ArithLogOp::Or => {
                                self.tyc
                                    .impose(left_key.concretizes_explicit(IAbstractType::Bool))?;
                                self.tyc
                                    .impose(right_key.concretizes_explicit(IAbstractType::Bool))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                            // Any x Any -> Bool COMPARATORS
                            ArithLogOp::Eq
                            | ArithLogOp::Lt
                            | ArithLogOp::Le
                            | ArithLogOp::Ne
                            | ArithLogOp::Ge
                            | ArithLogOp::Gt => {
                                self.tyc.impose(left_key.equate_with(right_key))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
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
                                self.tyc
                                    .impose(arg_key.concretizes_explicit(IAbstractType::Bool))?;

                                self.tyc
                                    .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                            }
                            //Num -> Num
                            ArithLogOp::Neg | ArithLogOp::BitNot => {
                                self.tyc
                                    .impose(arg_key.concretizes_explicit(IAbstractType::Numeric))?;

                                self.tyc.impose(term_key.equate_with(arg_key))?;
                            }
                            _ => unreachable!("All other operators have 2 given arguments"),
                        }
                    }
                    _ => unreachable!(),
                }
            }
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                // Bool for condition - check given in the second argument
                self.expression_infer(&*condition, Some(IAbstractType::Bool))?;
                let cons_key = self.expression_infer(&*consequence, None)?; // X
                let alt_key = self.expression_infer(&*alternative, None)?; // X
                                                                           //Bool x T x T -> T
                self.tyc
                    .impose(term_key.is_sym_meet_of(cons_key, alt_key))?;
            }

            ExpressionKind::Tuple(vec) => {
                let key_vec: Result<Vec<TcKey>, TcErr<IAbstractType>> = vec
                    .iter()
                    .map(|ex| self.expression_infer(ex, None))
                    .collect();
                let key_vec = key_vec?;
                let base_type = IAbstractType::Tuple(vec![IAbstractType::Any; vec.len()]);
                self.tyc.impose(term_key.concretizes_explicit(base_type))?;
                for (n, child_key_inferred) in key_vec.iter().enumerate() {
                    let n_key_given = self.tyc.get_child_key(term_key, n)?;
                    self.tyc
                        .impose(n_key_given.equate_with(*child_key_inferred))?;
                }
            }

            ExpressionKind::TupleAccess(expr, idx) => {
                let ex_key = self.expression_infer(expr, None)?;
                //Only Tuple case allowed for Field expression FIXME TODO
                //TODO enforce Vector type -> child access on any fails
                let accessed_child = self.tyc.get_child_key(ex_key, *idx)?;
                self.tyc.impose(term_key.equate_with(accessed_child))?;
            }

            ExpressionKind::Widen(inner, ty) => {
                let inner_expr_key = self.expression_infer(inner, None)?;
                let (upper_bound, type_bound) = match ty {
                    AnnotatedType::UInt(x) => {
                        (IAbstractType::UInteger(1), IAbstractType::UInteger(*x))
                    }
                    AnnotatedType::Int(x) => {
                        (IAbstractType::SInteger(1), IAbstractType::SInteger(*x))
                    }
                    AnnotatedType::Float(x) => (IAbstractType::Float(1), IAbstractType::Float(*x)),
                    _ => unimplemented!("Unsupported widen Type"),
                };
                let internal_key = self.tyc.new_term_key();
                self.tyc
                    .impose(internal_key.concretizes_explicit(type_bound.clone()))?;
                self.tyc.impose(internal_key.concretizes(inner_expr_key))?;

                self.tyc
                    .impose(inner_expr_key.concretizes_explicit(upper_bound))?;
                self.tyc.impose(term_key.has_exactly_type(type_bound))?;
            }
            ExpressionKind::Function {
                name,
                type_param,
                args,
            } => {
                dbg!("Function Infer");
                //transform Type into new internal types.
                let types_vec: Vec<IAbstractType> = type_param
                    .iter()
                    .map(|t| self.match_annotated_type(t))
                    .collect();
                // check for name in context
                let fun_decl = self.hir.func_declaration(name);
                dbg!(fun_decl);
                //Generics
                let generics: Vec<TcKey> = fun_decl
                    .generics
                    .iter()
                    .map(|gen| {
                        let gen_key: TcKey = self.tyc.new_term_key();
                        let ty = self.match_annotated_type(gen);
                        self.tyc
                            .impose(gen_key.concretizes_explicit(ty))
                            .map(|_| gen_key)
                    })
                    .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                for (t, gen) in types_vec.iter().zip(generics.iter()) {
                    self.tyc.impose(gen.concretizes_explicit(t.clone()))?;
                }
                //FOR: type.captures(generic)
                args.iter()
                    .zip(fun_decl.parameters.iter())
                    .map(|(arg, param)| {
                        //for (arg, param) in args.iter().zip(fun_decl.parameters.iter()) {
                        let p = self.replace_type(param, &generics)?;
                        let arg_key = self.expression_infer(&*arg, None)?;
                        self.tyc.impose(arg_key.concretizes(p))?;
                        Ok(arg_key)
                        //}
                    })
                    .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                let return_type = self.replace_type(&fun_decl.return_type, &generics)?;
                /*
                if name.name.contains("widen_") {
                    self.tyc.impose(
                        return_type.concretizes(
                            *arg_keys
                                .get(0)
                                .expect("build in widen function have exactly 1 argument"),
                        ),
                    )?;
                }
                */

                self.tyc.impose(term_key.concretizes(return_type))?;
            }
            ExpressionKind::ParameterAccess(current_stream, ix) => {
                let output: &Output = self
                    .hir
                    .outputs()
                    .nth(current_stream.out_ix())
                    .expect("StreamRef idx always valid");
                //let par_name = output.params[*ix].name.clone();
                let v = Variable {
                    name: output.name.clone() + "_" + &output.params[*ix].name,
                };
                let par_key = self.tyc.get_var_key(&v);
                dbg!(par_key);
                self.tyc.impose(term_key.equate_with(par_key))?;
            }
        };

        Ok(term_key)
        //Err(String::from("Error"))
    }

    fn replace_type(
        &mut self,
        at: &AnnotatedType,
        to: &[TcKey],
    ) -> Result<TcKey, TcErr<IAbstractType>> {
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
                self.tyc
                    .impose(replace_key.concretizes_explicit(self.match_annotated_type(at)))?;
                Ok(replace_key)
            }
        }
    }

    fn match_annotated_type(&self, t: &AnnotatedType) -> IAbstractType {
        match t {
            AnnotatedType::String => IAbstractType::TString,
            AnnotatedType::Int(x) => IAbstractType::SInteger(*x),
            AnnotatedType::Float(f) => IAbstractType::Float(*f),
            AnnotatedType::UInt(u) => IAbstractType::UInteger(*u),
            AnnotatedType::Bool => IAbstractType::Bool,
            AnnotatedType::Bytes => IAbstractType::UInteger(8),
            AnnotatedType::Option(op) => {
                IAbstractType::Option(self.match_annotated_type(&(**op)).into())
            }
            AnnotatedType::Tuple(v) => IAbstractType::Tuple(
                v.iter()
                    .map(|inner| self.match_annotated_type(inner))
                    .collect(),
            ),
            AnnotatedType::Numeric => IAbstractType::Numeric,
            AnnotatedType::Param(_, _) => todo!("currently handled externally"),
        }
    }

    fn match_const_literal(&self, lit: &ConstantLiteral) -> IAbstractType {
        dbg!(&lit);
        match lit {
            ConstantLiteral::Str(_) => IAbstractType::TString,
            //ConstantLiteral::Numeric(n, _post) => get_abstract_type_of_string_value(&n).unwrap(),
            ConstantLiteral::Bool(_) => IAbstractType::Bool,
            ConstantLiteral::Integer(_) => IAbstractType::Integer,
            ConstantLiteral::SInt(_) => IAbstractType::SInteger(1),
            ConstantLiteral::Float(_) => IAbstractType::Float(1),
        }
    }

    pub(crate) fn handle_error(
        &self,
        err: TcErr<IAbstractType>,
    ) -> <IAbstractType as Abstract>::Err {
        dbg!(&err);
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
                    None => format!(
                        "Invalid type bound enforced on {:?}: {}",
                        self.node_key.get_by_right(&key),
                        msg
                    ),
                    Some(k2) => format!(
                        "Invalid type bound enforced on {:?} by {:?}: {}",
                        self.node_key.get_by_right(&key).unwrap(),
                        self.node_key.get_by_right(&k2).unwrap(),
                        msg
                    ),
                }
            }
            TcErr::ExactTypeViolation(key, bound) => {
                primal_key = key;
                format!(
                    "Type Bound: {:?} incompatible with {:?}",
                    bound,
                    self.node_key.get_by_right(&key).unwrap()
                )
            }
            TcErr::ConflictingExactBounds(key, bound1, bound2) => {
                primal_key = key;
                format!(
                    "Incompatible bounds {:?} and {:?} applied on {:?}",
                    bound1,
                    bound2,
                    self.node_key.get_by_right(&key).unwrap()
                )
            }
        };
        if let Some(error_key_span) = self.key_span.get(&primal_key) {
            self.handler.error_with_span(
                "Stream inference error",
                error_key_span.clone(),
                Some(&msg),
            );
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
    use crate::rtltc::NodeId;
    use crate::value_types::IConcreteType;
    use crate::LolaTypeChecker;
    use front::common_ir::StreamReference;
    use front::hir::modes::IrExpression;
    use front::reporting::Handler;
    use front::RTLolaAst;
    use front::RTLolaHIR;
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
        let handler = front::reporting::Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst =
            match front::parse::parse(spec, &handler, front::FrontendConfig::default()) {
                Ok(s) => s,
                Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
            };
        let hir = front::hir::RTLolaHIR::<IrExpression>::transform_expressions(
            ast,
            &handler,
            &front::FrontendConfig::default(),
        );
        let mut na = front::analysis::naming::NamingAnalysis::new(
            &handler,
            front::FrontendConfig::default(),
        );
        //let mut dec = na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        TestBox { hir, handler }
    }

    fn complete_check(spec: &str) -> usize {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pacing_tt = ltc.pacing_type_infer().unwrap();
        ltc.value_type_infer(pacing_tt);
        test_box.handler.emitted_errors()
    }

    fn check_value_type(spec: &str) -> (TestBox, HashMap<NodeId, IConcreteType>) {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pacing_tt = ltc.pacing_type_infer().expect("Expected valid pacing type");
        let tt_result = ltc.value_type_infer(pacing_tt);
        if let Err(ref e) = tt_result {
            eprintln!("{}", e.clone());
        }
        assert!(
            tt_result.is_ok(),
            "Expect Valid Input - Value Type check failed"
        );
        let tt = tt_result.expect("ensured by assertion");
        (test_box, tt)
    }

    fn check_expect_error(spec: &str) -> TestBox {
        let test_box = setup_hir(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.hir, &test_box.handler);
        let pt = ltc.pacing_type_infer().expect("expect valid pacing input");
        let tt_result = ltc.value_type_infer(pt);
        dbg!(&tt_result);
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
        assert_eq!(
            result_map[&NodeId::SRef(output_sr)],
            IConcreteType::Integer8
        );
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn direct_widening() {
        let spec = "input i: Int8\noutput o :Int32 := widen<Int32>(i)";
        let (tb, result_map) = check_value_type(spec);
        let input_sr = tb.hir.get_input_with_name("i").unwrap().sr;
        let output_sr = tb.hir.get_output_with_name("o").unwrap().sr;
        assert_eq!(result_map[&NodeId::SRef(input_sr)], IConcreteType::Integer8);
        assert_eq!(
            result_map[&NodeId::SRef(output_sr)],
            IConcreteType::Integer32
        );
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
        assert_eq!(
            result_map[&NodeId::SRef(input_i_id)],
            IConcreteType::Integer8
        );
        assert_eq!(
            result_map[&NodeId::SRef(input_i1_id)],
            IConcreteType::Integer16
        );
        assert_eq!(
            result_map[&NodeId::SRef(output_id)],
            IConcreteType::Integer16
        );
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    //#[ignore] //PARSER ERROR TODO after typecheck ,during error reporting
    fn parametric_access_default() {
        let spec = "output i(a: Int8, b: Bool): Int8 @1Hz := if b then a else 0 \n output o := i(1,false).offset(by:-1).defaults(to: 42)";
        let (tb, result_map) = check_value_type(spec);
        let o2_sr = tb.output("i");
        let o1_id = tb.output("o");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(o1_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(o2_sr)], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_x() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz := 1";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&NodeId::SRef(output_sr)],
            IConcreteType::Integer8
        );
    }

    #[test]
    fn parametric_declaration_param_infer() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz := a";
        let (tb, result_map) = check_value_type(spec);
        let output_sr = tb.output("x");
        assert_eq!(0, complete_check(spec));
        assert_eq!(0, tb.handler.emitted_errors());
        assert_eq!(
            result_map[&NodeId::SRef(output_sr)],
            IConcreteType::UInteger8
        );
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz := 1 output y @1Hz := x(1, false)";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.output("x");
        let output_2_id = tb.output("y");
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&NodeId::SRef(output_id)],
            IConcreteType::Integer8
        );
        assert_eq!(
            result_map[&NodeId::SRef(output_2_id)],
            IConcreteType::Integer8
        );
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
            let (tb, result_map) = check_value_type(spec);
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
        let spec =
            "constant c: Int32 := 1\n constant d: Int8 := 2\noutput o @1Hz := c + widen<Int32>(d)";
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
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 @1Hz { extend in } := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Bool);
    }

    #[test]
    //#[ignore] // paramertic streams need new design after syntax revision
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 @1Hz { extend in } := 3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a(b: Bool): Int8 @1Hz {close in} := 3";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let in_id = tb.input("in");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(in_id)], IConcreteType::Bool);
    }

    #[test]
    fn test_terminate_type_faulty() {
        //Close condition non boolean type
        let spec = "input in: Int8\n output a(b: Bool): Int8 @1Hz {close in} := 3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
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
    fn test_param_spec_faulty() {
        let spec = "output a(p1: Int8): Int8 @1Hz:= 3 output b: Int8 := a(true)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
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
    fn test_param_inferred_conflicting() {
        let spec = "input i: Int8, j: UInt8 output x(param): Int8 := i output y: Int8 := x(i) output z: Int8 := x(j)";
        let tb = check_expect_error(spec);
        assert_eq!(complete_check(spec), 1);
        //assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8)); //TODO
    }

    #[test]
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
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
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
        assert_eq!(
            result_map[&NodeId::SRef(out_id)],
            IConcreteType::Option(IConcreteType::Integer8.into())
        );
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
            IConcreteType::Integer8, //Changed to 16 FIXME
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

    #[test]
    #[ignore] //currently rejected in new hir
    fn test_window_invalid_duration() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 0s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: -3s, using: Σ)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    #[ignore] // uint to int cast currently not allowed
    fn test_aggregation_implicit_cast() {
        let spec =
                            "input in: UInt8\n output out: Int16 @5Hz := in.aggregate(over_exactly: 3s, using: Σ).defaults(to: 5)";
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
    //#[ignore] //TODO real time offsets based on pacing type analysis
    fn test_rt_offset() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.output("a");
        let out_id2 = tb.output("b");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer8);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer8);
    }

    #[test]
    fn test_rt_offset_regression() {
        let spec = "output a @10Hz := a.offset(by: -100ms).defaults(to: 0) + 1";
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
        let out_id2 = tb.output("x2");
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&NodeId::SRef(out_id)], IConcreteType::Integer32);
        assert_eq!(result_map[&NodeId::SRef(out_id2)], IConcreteType::Integer32);
    }

    #[test]
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-1s].defaults(to: 0)";
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
