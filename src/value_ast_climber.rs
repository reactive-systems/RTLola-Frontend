use super::*;
extern crate regex;

use crate::pacing_types::ConcretePacingType;
use crate::value_types::IAbstractType;
use bimap::BiMap;
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Constant, Input, Output, Trigger};
use front::ast::{Expression, LitKind, Type};
use front::ast::{ExpressionKind, Parameter, TypeKind};
use front::parse::{NodeId, Span};
use front::reporting::{Handler, LabeledSpan};
use front::ty::{TypeConstraint, ValueTy};
use rusttyc::types::Abstract;
use rusttyc::{TcErr, TcKey, TypeChecker};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct ValueContext<'a> {
    pub(crate) tyc: TypeChecker<IAbstractType, Variable>,
    pub(crate) decl: DeclarationTable,
    //Map assumes uniqueness of Ast and Tc ids
    pub(crate) node_key: BiMap<NodeId, TcKey>,
    pub(crate) key_span: HashMap<TcKey, Span>,
    pub(crate) handler: &'a Handler,
    pub(crate) pacing_tt: HashMap<NodeId, ConcretePacingType>,
}

impl<'a> ValueContext<'a> {
    pub fn new(
        ast: &RTLolaAst,
        decl: DeclarationTable,
        handler: &'a Handler,
        pacing_tt: HashMap<NodeId, ConcretePacingType>,
    ) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = BiMap::new();
        let mut key_span = HashMap::new();

        for input in &ast.inputs {
            let key = tyc.get_var_key(&Variable {
                name: input.name.name.clone(),
            });
            node_key.insert(input.id, key);
            key_span.insert(key, input.span);
        }

        for cons in &ast.constants {
            let key = tyc.get_var_key(&Variable {
                name: cons.name.name.clone(),
            });
            node_key.insert(cons.id, key);
            key_span.insert(key, cons.span);
        }

        for out in &ast.outputs {
            let key = tyc.get_var_key(&Variable {
                name: out.name.name.clone(),
            });
            dbg!(key);
            node_key.insert(out.id, key);
            key_span.insert(key, out.span);
        }

        for (ix, tr) in ast.trigger.iter().enumerate() {
            let n = match &tr.name {
                Some(ident) => ident.name.clone() + "_" + &ix.to_string(),
                None => format!("trigger_{}", ix),
            };
            let key = tyc.get_var_key(&Variable { name: n });
            node_key.insert(tr.id, key);
            key_span.insert(key, tr.span);
        }

        ValueContext {
            tyc,
            decl,
            node_key,
            key_span,
            handler,
            pacing_tt,
        }
    }

    pub fn input_infer(&mut self, input: &Input) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = *self
            .node_key
            .get_by_left(&input.id)
            .expect("Added in constructor");
        //Annotated Type

        let annotated_type_replaced = self.type_kind_match(&input.ty);
        //can skip any case as type must be provided
        self.tyc
            .impose(term_key.has_exactly_type(annotated_type_replaced))?;

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
        Ok(term_key)
    }

    pub fn constant_infer(&mut self, cons: &Constant) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = *self
            .node_key
            .get_by_left(&cons.id)
            .expect("Added in constructor");
        //Annotated Type
        if let Some(t) = &cons.ty {
            let annotated_type_replaced = self.type_kind_match(t);
            if annotated_type_replaced != IAbstractType::Any {
                self.tyc
                    .impose(term_key.has_exactly_type(annotated_type_replaced))?;
            }
        }
        //Type from Literal
        let lit_type = self.match_lit_kind(cons.literal.kind.clone());
        self.tyc.impose(term_key.concretizes_explicit(lit_type))?;

        self.node_key.insert(cons.id, term_key);
        Ok(term_key)
    }

    pub fn output_infer(&mut self, out: &Output) -> Result<TcKey, TcErr<IAbstractType>> {
        let out_key = *self
            .node_key
            .get_by_left(&out.id)
            .expect("Added in constructor");

        let annotated_type_replaced = self.type_kind_match(&out.ty);
        dbg!(&annotated_type_replaced);
        if let IAbstractType::Any = annotated_type_replaced {
        } else {
            self.tyc
                .impose(out_key.has_exactly_type(annotated_type_replaced))?;
        }

        let mut param_types = Vec::new();
        for param in &out.params {
            let param_key = self.tyc.get_var_key(&Variable {
                name: out.name.name.clone() + "_" + &param.name.name.clone(),
            });
            dbg!(param_key);
            self.node_key.insert(param.id, param_key);
            self.key_span.insert(param_key, param.span);

            let t = self.type_kind_match(&param.ty);
            self.tyc.impose(param_key.concretizes_explicit(t))?;
            param_types.push(param_key);
        }

        dbg!(&out.template_spec);
        if let Some(template_spec) = &out.template_spec {
            if let Some(inv) = &template_spec.inv {
                //chek target exression type matches parameter type
                let target_expr_key = self.expression_infer(&inv.target, None)?;
                //TODO
            }
            if let Some(close) = &template_spec.ter {
                self.expression_infer(&close.target, Some(IAbstractType::Bool))?;
            }
            if let Some(ext) = &template_spec.ext {
                self.expression_infer(&ext.target, Some(IAbstractType::Bool))?;
            }
        }

        let expression_key = self.expression_infer(&out.expression, None)?;

        self.tyc.impose(out_key.equate_with(expression_key))?;
        Ok(out_key)
    }

    pub fn trigger_infer(&mut self, tr: &Trigger) -> Result<TcKey, TcErr<IAbstractType>> {
        let tr_key = *self
            .node_key
            .get_by_left(&tr.id)
            .expect("Added in constructor");
        let expression_key = self.expression_infer(&tr.expression, Some(IAbstractType::Bool))?;
        self.tyc.impose(tr_key.concretizes(expression_key))?;
        Ok(tr_key)
    }

    fn expression_infer(
        &mut self,
        exp: &Expression,
        target_type: Option<IAbstractType>,
    ) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        dbg!(term_key, "added to key span map");
        self.key_span.insert(term_key, exp.span);
        if let Some(t) = target_type {
            self.tyc.impose(term_key.concretizes_explicit(t))?;
        }
        dbg!(&exp.kind);
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = self.match_lit_kind(lit.kind.clone());
                //dbg!(&literal_type);
                self.tyc
                    .impose(term_key.concretizes_explicit(literal_type))?;
            }
            ExpressionKind::Ident(_) => {
                let decl = &self.decl[&exp.id];
                let node_id = match decl {
                    Declaration::Const(c) => c.id,
                    Declaration::Out(out) => out.id,
                    Declaration::ParamOut(param) => param.id,
                    Declaration::In(input) => input.id,
                    Declaration::Param(p) => p.id,
                    Declaration::Type(_) | Declaration::Func(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                };
                let key = self
                    .node_key
                    .get_by_left(&node_id)
                    .expect("Value should be contained");
                self.tyc.impose(term_key.equate_with(*key))?;
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infer(&*ex, None)?;
                match kind {
                    Sync => {
                        //Sync access just returns the stream type
                        self.tyc.impose(term_key.concretizes(ex_key))?;
                    }
                    Optional | Hold => {
                        //Optional and Hold return Option<X> Type
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Option(
                                IAbstractType::Any.into(),
                            )))?;
                        let inner_key = self.tyc.get_child_key(term_key, 1)?;
                        self.tyc.impose(ex_key.equate_with(inner_key))?;
                    }
                };
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infer(&*ex, None)?; //Option<X>
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
            ExpressionKind::Offset(expr, offset) => {
                let ex_key = self.expression_infer(&*expr, None)?; // X
                                                                   //Want build: Option<X>

                match offset {
                    front::ast::Offset::Discrete(i) => {
                        if 0 == *i {
                            //Offset of 0 does not return an optional type
                            self.tyc.impose(term_key.equate_with(ex_key))?;
                        /*} else if *i > 0 { // unsure if correct - some tests fail then
                        return Err(TcErr::Bound(
                            term_key,
                            None,
                            "Found positive offset - not yet supported".to_string(),
                        ));
                        */
                        } else {
                            self.tyc.impose(term_key.concretizes_explicit(
                                IAbstractType::Option(IAbstractType::Any.into()),
                            ))?;
                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                            self.tyc.impose(ex_key.equate_with(inner_key))?;
                        }
                    }
                    front::ast::Offset::RealTime(_r, _unit) => {
                        //if periode < offset -> optinal
                        //
                        /*
                        if *r.numer() == 0 {
                            self.tyc.impose(term_key.equate_with(ex_key))?;
                        } else {
                            use crate::pacing_types::ConcretePacingType::*;
                            let current_ratio: &ConcretePacingType = &self.pacing_tt[&exp.id];
                            let target_ratio: &ConcretePacingType = &self.pacing_tt[&expr.id];
                            /* //TODO FIXME
                            match (current_ratio, target_ratio) {
                                (FixedPeriodic(f1),FixedPeriodic(f2)) => f1.
                            }
                            */
                        }
                        */
                        //TODO there are no realtime offsets so far
                        unimplemented!("RealTime offset not yet supported in Value Type inference")
                    }
                }
            }
            ExpressionKind::SlidingWindowAggregation {
                expr,
                duration,
                wait,
                aggregation: aggr,
            } => {
                let ex_key = self.expression_infer(&*expr, None)?;
                let duration_key = self.expression_infer(&*duration, None)?;

                self.tyc
                    .impose(duration_key.concretizes_explicit(IAbstractType::Numeric))?;

                use front::ast::WindowOperation;
                match aggr {
                    //Min|Max|Avg <T:Num> T -> Option<T>
                    WindowOperation::Min | WindowOperation::Max | WindowOperation::Average => {
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Option(
                                IAbstractType::Any.into(),
                            )))?;
                        let inner_key = self.tyc.get_child_key(term_key, 0)?;
                        self.tyc.impose(inner_key.equate_with(ex_key))?;
                    }
                    //Count: Any -> uint
                    WindowOperation::Count => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Any))?;
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::UInteger(1)))?;
                    }
                    //integral :T <T:Num> -> T
                    //integral : T <T:Num> -> Float   <-- currently used
                    WindowOperation::Integral => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Numeric))?; //TODO maybe numeric
                        if *wait {
                            self.tyc.impose(term_key.concretizes_explicit(
                                IAbstractType::Option(IAbstractType::Any.into()),
                            ))?;
                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                            //self.tyc.impose(inner_key.equate_with(ex_key))?;
                            self.tyc
                                .impose(inner_key.concretizes_explicit(IAbstractType::Float(1)))?;
                        } else {
                            //self.tyc.impose(term_key.concretizes(ex_key))?;
                            self.tyc
                                .impose(term_key.concretizes_explicit(IAbstractType::Float(1)))?;
                        }
                    }
                    //all others :T <T:Num> -> T
                    WindowOperation::Sum | WindowOperation::Product => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Numeric))?;
                        if *wait {
                            self.tyc.impose(term_key.concretizes_explicit(
                                IAbstractType::Option(IAbstractType::Any.into()),
                            ))?;
                            let inner_key = self.tyc.get_child_key(term_key, 0)?;
                            self.tyc.impose(inner_key.equate_with(ex_key))?;
                        } else {
                            self.tyc.impose(term_key.concretizes(ex_key))?;
                        }
                    }
                    //bool -> bool
                    WindowOperation::Conjunction | WindowOperation::Disjunction => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Bool))?;
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                    }
                }
            }
            //TODO
            ///// implicit widening requieres join operand
            // a + b -> c c = meet(a,b) then equate a and b with join(a,b) //FIXME
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infer(&*left, None)?; // X
                let right_key = self.expression_infer(&*right, None)?; // X

                //let param_key = self.tyc.new_term_key();

                use front::ast::BinOp;
                match op {
                    // <T:Num> T x T -> T
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                        /*
                        self.tyc.impose(param_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc.impose(param_key.concretizes(left_key))?;
                        self.tyc.impose(param_key.concretizes(right_key))?;
                        self.tyc.impose(term_key.concretizes(param_key))?;
                        */

                        self.tyc
                            .impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc
                            .impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;

                        self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                        self.tyc.impose(term_key.equate_with(left_key))?;
                        self.tyc.impose(term_key.equate_with(right_key))?;
                    }
                    // Bool x Bool -> Bool
                    BinOp::And | BinOp::Or => {
                        self.tyc
                            .impose(left_key.concretizes_explicit(IAbstractType::Bool))?;
                        self.tyc
                            .impose(right_key.concretizes_explicit(IAbstractType::Bool))?;

                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                    }
                    // Num x Num -> Num
                    BinOp::BitXor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                        self.tyc
                            .impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc
                            .impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;

                        self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                        self.tyc.impose(term_key.equate_with(left_key))?;
                        self.tyc.impose(term_key.equate_with(right_key))?;
                    }
                    // Any x Any -> Bool COMPARATORS
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                        //self.tyc
                        //    .impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                        //self.tyc
                        //    .impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc.impose(left_key.equate_with(right_key))?;

                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                    }
                }
            }
            ExpressionKind::Unary(op, expr) => {
                let ex_key = self.expression_infer(&*expr, None)?; // expr

                use front::ast::UnOp;
                match op {
                    //Num -> Num
                    UnOp::BitNot | UnOp::Neg => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Numeric))?;

                        self.tyc.impose(term_key.equate_with(ex_key))?;
                    }
                    // Bool -> Bool
                    UnOp::Not => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Bool))?;

                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::Bool))?;
                    }
                }
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                // Bool for condition - check given in the second argument
                self.expression_infer(&*cond, Some(IAbstractType::Bool))?;
                let cons_key = self.expression_infer(&*cons, None)?; // X
                let alt_key = self.expression_infer(&*alt, None)?; // X
                                                                   //Bool x T x T -> T

                self.tyc
                    .impose(term_key.is_sym_meet_of(cons_key, alt_key))?;
                //self.tyc.impose(cons_key.equate_with(alt_key))?;
            }
            ExpressionKind::MissingExpression => unreachable!(),
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
            ExpressionKind::Field(expr, ident) => {
                let ex_key = self.expression_infer(expr, None)?;
                //Only Tuple case allowed for Field expression FIXME TODO
                let n: usize = ident.name.parse().expect("checked in AST verifier");
                //TODO enforce Vector type -> child access on any fails
                let accessed_child = self.tyc.get_child_key(ex_key, n)?;
                self.tyc.impose(term_key.equate_with(accessed_child))?;
            }
            ExpressionKind::Method(_, _, _, _) => unimplemented!("TODO"),
            ExpressionKind::Function(name, types, args) => {
                dbg!("Function Infer");
                //transform Type into new internal types.
                let types_vec: Vec<IAbstractType> =
                    types.iter().map(|t| self.type_kind_match(&t)).collect();
                // check for name in context
                let decl = self
                    .decl
                    .get(&exp.id)
                    .expect("declaration checked by naming analysis")
                    .clone();
                match &decl {
                    Declaration::Func(fun_decl) => {
                        dbg!(fun_decl);
                        //Generics
                        let generics: Vec<TcKey> = fun_decl
                            .generics
                            .iter()
                            .map(|gen| {
                                let gen_key: TcKey = self.tyc.new_term_key();
                                dbg!(gen_key);
                                let rusttyc_result = match &gen {
                                    ValueTy::Constr(tc) => {
                                        let cons = match_constraint(tc);
                                        self.tyc.impose(gen_key.concretizes_explicit(cons))
                                    }
                                    _ => unreachable!("function declarations are not user-definable and currently, only constraints are allowed for generic types"),
                                };
                                rusttyc_result.map(|_| gen_key)
                            })
                            .collect::<Result<Vec<TcKey>, TcErr<IAbstractType>>>()?;

                        for (t, gen) in types_vec.iter().zip(generics.iter()) {
                            let t_key = self.tyc.new_term_key();

                            self.tyc.impose(t_key.concretizes_explicit(t.clone()))?;
                            self.tyc.impose(t_key.concretizes(*gen))?;
                        }
                        //FOR: type.captures(generic)
                        let arg_keys_result: Result<Vec<TcKey>, TcErr<IAbstractType>> = args
                            .iter()
                            .zip(fun_decl.parameters.iter())
                            .map(|(arg, param)| {
                                //for (arg, param) in args.iter().zip(fun_decl.parameters.iter()) {
                                let p = self.replace_type(param, &generics)?;
                                let arg_key = self.expression_infer(&*arg, None)?;
                                self.tyc.impose(arg_key.concretizes(p))?;
                                Ok(arg_key)
                                //}
                            })
                            .collect();
                        let arg_keys = arg_keys_result?;

                        let return_type = self.replace_type(&fun_decl.return_type, &generics)?;
                        if name.name.name.contains("widen_") {
                            self.tyc.impose(
                                return_type.concretizes(
                                    *arg_keys
                                        .get(0)
                                        .expect("build in widen function have exactly 1 argument"),
                                ),
                            )?;
                        }

                        self.tyc.impose(term_key.concretizes(return_type))?;
                    }
                    Declaration::ParamOut(out) => {
                        // output a(i: int, flag: bool) @1hz spawn (input_int , input_bool) if input_bool term b := if flag then i else 0
                        // output c @1hz := a(input_int, false)
                        let params: &[Rc<Parameter>] = out.params.as_slice();
                        let param_out_tckey = self.tyc.get_var_key(&Variable {
                            name: out.name.name.clone(),
                        });

                        let param_keys: Vec<TcKey> = out
                            .params
                            .iter()
                            .map(|p| {
                                self.tyc.get_var_key(&Variable {
                                    name: out.name.name.clone() + "_" + &p.name.name.clone(),
                                })
                            })
                            .collect();

                        let param_types: Vec<IAbstractType> =
                            params.iter().map(|p| self.type_kind_match(&p.ty)).collect();
                        //dbg!(&param_types);
                        for ((arg, param_t), p_key) in
                            args.iter().zip(param_types.iter()).zip(param_keys.iter())
                        {
                            let arg_key = self.expression_infer(&*arg, Some(param_t.clone()))?;
                            /*
                            dbg!(arg, param_t);
                            self.node_key.insert(arg.id,arg_key);
                            dbg!(arg.id,arg_key,p_key);
                            */
                            self.tyc.impose(p_key.equate_with(arg_key))?;
                            self.tyc
                                .impose(p_key.concretizes_explicit(param_t.clone()))?;

                            self.tyc
                                .impose(arg_key.concretizes_explicit(param_t.clone()))?;
                        }

                        let annotated_type = self.type_kind_match(&out.ty);
                        //dbg!(&annotated_type);
                        self.tyc
                            .impose(term_key.concretizes_explicit(annotated_type))?;
                        self.tyc.impose(term_key.concretizes(param_out_tckey))?;
                    }
                    _ => unreachable!("ensured by naming analysis"),
                };
            }
            ExpressionKind::ParenthesizedExpression(_, _, _) => unimplemented!(),
        };

        self.node_key.insert(exp.id, term_key);
        Ok(term_key)
        //Err(String::from("Error"))
    }

    fn replace_type(&mut self, vt: &ValueTy, to: &[TcKey]) -> Result<TcKey, TcErr<IAbstractType>> {
        match vt {
            &ValueTy::Param(idx, _) => Ok(to[idx as usize]),
            ValueTy::Option(o) => {
                let op_key = self.tyc.new_term_key();
                let inner = self.replace_type(o, to)?;
                self.tyc.impose(
                    op_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())),
                )?;
                let inner_tc = self.tyc.get_child_key(op_key, 0)?;
                self.tyc.impose(inner_tc.equate_with(inner))?;
                Ok(op_key)
            }
            ValueTy::Constr(c) => {
                let key = self.tyc.new_term_key();
                dbg!(key);
                self.tyc
                    .impose(key.concretizes_explicit(match_constraint(c)))
                    .map(|_| key)
            }
            _ if vt.is_primitive() => {
                let key = self.tyc.new_term_key();
                self.tyc
                    .impose(key.concretizes_explicit(self.value_type_match(vt)))
                    .map(|_| key)
            }
            _ => unreachable!("replace for {}", vt),
        }
    }

    fn type_kind_match(&self, t: &Type) -> IAbstractType {
        let kind = &t.kind;
        match kind {
            TypeKind::Simple(_) => {
                let decl = self.decl[&t.id].clone();
                if let Declaration::Type(ty) = decl {
                    self.value_type_match(&ty)
                } else {
                    unreachable!("ensured by naming analysis")
                }
            }
            TypeKind::Tuple(v) => {
                IAbstractType::Tuple(v.iter().map(|t| self.type_kind_match(&t)).collect())
            }
            TypeKind::Optional(op) => IAbstractType::Option(self.type_kind_match(&op).into()),
            TypeKind::Inferred => IAbstractType::Any,
        }
    }

    fn value_type_match(&self, vt: &ValueTy) -> IAbstractType {
        match vt {
            ValueTy::Bool => IAbstractType::Bool,
            ValueTy::Int(i) => {
                use front::ty::IntTy;
                match i {
                    IntTy::I8 => IAbstractType::SInteger(8),
                    IntTy::I16 => IAbstractType::SInteger(16),
                    IntTy::I32 => IAbstractType::SInteger(32),
                    IntTy::I64 => IAbstractType::SInteger(64),
                }
            }
            ValueTy::UInt(u) => {
                use front::ty::UIntTy;
                match u {
                    UIntTy::U8 => IAbstractType::UInteger(8),
                    UIntTy::U16 => IAbstractType::UInteger(16),
                    UIntTy::U32 => IAbstractType::UInteger(32),
                    UIntTy::U64 => IAbstractType::UInteger(64),
                }
            }
            ValueTy::Float(f) => {
                use front::ty::FloatTy;
                match f {
                    FloatTy::F16 => IAbstractType::Float(16),
                    FloatTy::F32 => IAbstractType::Float(32),
                    FloatTy::F64 => IAbstractType::Float(64),
                }
            }
            ValueTy::String => IAbstractType::TString,
            ValueTy::Bytes => unimplemented!(),
            ValueTy::Tuple(vec) => {
                IAbstractType::Tuple(vec.iter().map(|t| self.value_type_match(t)).collect())
            }
            ValueTy::Option(o) => IAbstractType::Option(self.value_type_match(&**o).into()),
            ValueTy::Infer(_) => unreachable!(),
            ValueTy::Constr(c) => match_constraint(c),
            ValueTy::Param(_, _) => {
                unimplemented!("Param case should only be addressed in replace_type(...)")
            }
            ValueTy::Error => unreachable!("Error should be checked before hand"),
        }
    }

    fn match_lit_kind(&self, lit: LitKind) -> IAbstractType {
        dbg!(&lit);
        match lit {
            LitKind::Str(_) | LitKind::RawStr(_) => IAbstractType::TString,
            //TODO post unused
            LitKind::Numeric(n, _post) => get_abstract_type_of_string_value(&n).unwrap(),
            LitKind::Bool(_) => IAbstractType::Bool,
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
        let error_key_span = self.key_span[&primal_key];
        self.handler.error_with_span(
            "Input inference error",
            LabeledSpan::new(error_key_span, &msg, true),
        );
        msg
    }
}

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

fn match_constraint(cons: &TypeConstraint) -> IAbstractType {
    //TODO
    let r = match cons {
        TypeConstraint::Numeric => IAbstractType::Numeric,
        TypeConstraint::SignedInteger => IAbstractType::SInteger(1),
        TypeConstraint::UnsignedInteger => IAbstractType::UInteger(1),
        TypeConstraint::FloatingPoint => IAbstractType::Float(1),
        TypeConstraint::Integer => IAbstractType::Integer,
        TypeConstraint::Equatable => unimplemented!(),
        TypeConstraint::Comparable => unimplemented!(),
        TypeConstraint::Unconstrained => IAbstractType::Any,
    };
    dbg!(&r);
    r
}

#[cfg(test)]
mod value_type_tests {
    use crate::value_types::IConcreteType;
    use crate::LolaTypeChecker;
    use front::analysis::naming::Declaration;
    use front::parse::NodeId;
    use front::parse::SourceMapper;
    use front::reporting::Handler;
    use front::RTLolaAst;
    use std::collections::HashMap;
    use std::path::PathBuf;

    struct TestBox {
        pub spec: RTLolaAst,
        pub dec: HashMap<NodeId, Declaration>,
        pub handler: Handler,
    }

    fn setup_ast(spec: &str) -> TestBox {
        let handler = front::reporting::Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let spec: RTLolaAst =
            match front::parse::parse(spec, &handler, front::FrontendConfig::default()) {
                Ok(s) => s,
                Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
            };
        let mut na = front::analysis::naming::NamingAnalysis::new(
            &handler,
            front::FrontendConfig::default(),
        );
        let mut dec = na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        TestBox { spec, dec, handler }
    }

    fn complete_check(spec: &str) -> usize {
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
        let pacing_tt = ltc.pacing_type_infer().unwrap();
        ltc.value_type_infer(pacing_tt);
        test_box.handler.emitted_errors()
    }

    fn check_value_type(spec: &str) -> (TestBox, HashMap<NodeId, IConcreteType>) {
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
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
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
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
        let input_id = tb.spec.inputs[0].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(result_map[&input_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn direct_widening() {
        let spec = "input i: Int8\noutput o :Int32 := widen_signed(i)";
        let (tb, result_map) = check_value_type(spec);
        let input_id = tb.spec.inputs[0].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(result_map[&input_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_id], IConcreteType::Integer32);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn integer_addition() {
        let spec = "input i: Int8\ninput i1: Int16\noutput o := widen_signed(i) + i1";
        let (tb, result_map) = check_value_type(spec);
        let input_i_id = tb.spec.inputs[0].id;
        let input_i1_id = tb.spec.inputs[1].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(result_map[&input_i_id], IConcreteType::Integer8);
        assert_eq!(result_map[&input_i1_id], IConcreteType::Integer16);
        assert_eq!(result_map[&output_id], IConcreteType::Integer16);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn parametric_access_default() {
        let spec = "output i(a: Int8, b: Bool): Int8 @1Hz := if b then a else 0\noutput o := i(1,false)[-1].defaults(to: 42)";
        let (tb, result_map) = check_value_type(spec);
        let o2_id = tb.spec.outputs[1].id;
        let o1_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&o1_id], IConcreteType::Integer8);
        assert_eq!(result_map[&o1_id], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_x() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz := 1";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_infer() {
        let spec = "output x(a: UInt8, b: Bool) @1Hz := a";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(0, tb.handler.emitted_errors());
        assert_eq!(result_map[&output_id], IConcreteType::UInteger8);
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 @1Hz := 1 output y @1Hz := x(1, false)";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        let output_2_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_2_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Float32);
    }

    #[test]
    #[ignore] //TODO float parse to float16 invalid due to exact given bound application
    fn simple_const_float16() {
        let spec = "constant c: Float16 := 2.1";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Float32);
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_valid_coersion() {
        //TODO does not check output type, only validity
        for spec in &[
            "constant c: Int8 := 1\noutput o: Int32 @1Hz:= widen_signed(c)",
            "constant c: UInt16 := 1\noutput o: UInt64 @1Hz:= widen_unsigned(c)",
            "constant c: Float32 := 1.0\noutput o: Float64 @1Hz:= widen_float(c)",
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
        let spec = "constant c: Int32 := 1\noutput o: Int8 @1Hz := widen_signed(c)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn simple_explicit_widening() {
        let spec =
            "constant c: Int32 := 1\n constant d: Int8 := 2\noutput o @1Hz := c + widen_signed(d)";
        let (tb, result_map) = check_value_type(spec);
        let c_id = tb.spec.constants[0].id;
        let d_id = tb.spec.constants[1].id;
        let o_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&c_id], IConcreteType::Integer32);
        assert_eq!(result_map[&d_id], IConcreteType::Integer8);
        assert_eq!(result_map[&o_id], IConcreteType::Integer32);
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger false";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.spec.trigger[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&tr_id], IConcreteType::Bool);
    }

    #[test]
    fn simple_trigger_message() {
        let spec = "trigger false \"alert always\"";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.spec.trigger[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&tr_id], IConcreteType::Bool);
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
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_binary_input() {
        let spec = "input i: Int8\noutput o: Int8 := 3 + i";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_unary() {
        let spec = "output o @1Hz:= !false \n\
                           output u: Bool @1Hz:= !false";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let out_id_2 = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Bool);
        assert_eq!(result_map[&out_id_2], IConcreteType::Bool);
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
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_ite_compare() {
        let spec = "output e :Int8 @1Hz := if 1 == 0 then 0 else -1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o @1Hz := if !false then 1.3 else -2.0";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Float32);
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Bool);
        assert_eq!(result_map[&in_id], IConcreteType::TString);
    }

    #[test]
    fn test_underspecified_type() {
        //Default for num literals applied
        let spec = "output o @1Hz := 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        let res_type = &result_map[&out_id];
        assert!(*res_type == IConcreteType::Integer32 || *res_type == IConcreteType::Integer64);
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
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
        let out_id = tb.spec.outputs[0].id;
        let out2_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&out2_id], IConcreteType::UInteger8);
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&in_id], IConcreteType::Bool);
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&in_id], IConcreteType::Bool);
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
        let out_id = tb.spec.outputs[0].id;
        let out2_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out2_id], IConcreteType::Integer8);
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
        let out_id = tb.spec.outputs[0].id;
        let out2_id = tb.spec.outputs[1].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out2_id], IConcreteType::Integer8);
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
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&out_id],
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(
            result_map[&in_id],
            IConcreteType::Tuple(vec![IConcreteType::Integer8, IConcreteType::Bool])
        );
        assert_eq!(result_map[&out_id], IConcreteType::Bool);
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
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(
            result_map[&out_id],
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
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        //assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&out_id], IConcreteType::UInteger8);
    }

    #[test]
    fn test_tuple_of_tuples() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Int16 := widen_signed(in[0].0)";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        let input_type = IConcreteType::Tuple(vec![
            IConcreteType::Integer8, //Changed to 16 FIXME
            IConcreteType::Tuple(vec![IConcreteType::UInteger8, IConcreteType::Bool]),
        ]);
        assert_eq!(result_map[&in_id], input_type);
        assert_eq!(result_map[&out_id], IConcreteType::Integer16);
    }

    #[test]
    #[ignore] //tuple with variable length are not supported TODO
    fn test_tuple_of_tuples2() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Bool := in.1.1";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        let input_type = IConcreteType::Tuple(vec![
            IConcreteType::Integer8,
            IConcreteType::Tuple(vec![IConcreteType::UInteger8, IConcreteType::Bool]),
        ]);
        assert_eq!(result_map[&in_id], input_type);
        assert_eq!(result_map[&out_id], IConcreteType::Bool);
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 @5Hz:= in.aggregate(over: 3s, using: Σ)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Integer64);
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
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
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&out_id], IConcreteType::Integer16);
    }

    #[test]
    #[ignore] //symmetric type relation extends input int8 to float
    fn test_aggregation_implicit_cast2() {
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: avg).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Float32);
    }

    #[test]
    //#[ignore] //symmetric type relation extends input int8 to float
    fn test_aggregation_implicit_cast3() {
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Float32);
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
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::UInteger8);
        assert_eq!(result_map[&out_id], IConcreteType::Float32);
        let spec =
            "input in: Int8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out_id], IConcreteType::Float32);
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := widen_float(velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0))";
        let (tb, result_map) = check_value_type(spec);
        let in_id = tb.spec.inputs[0].id;
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&in_id], IConcreteType::Float32);
        assert_eq!(result_map[&out_id], IConcreteType::Float64);
    }

    #[test]
    #[ignore] //TODO real time offsets based on pacing type analysis
    fn test_rt_offset() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        let (tb, result_map) = check_value_type(spec);
        let out1_id = tb.spec.outputs[0].id;
        let out2_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out1_id], IConcreteType::Integer8);
        assert_eq!(result_map[&out2_id], IConcreteType::Integer8);
    }

    /* //TODO
    #[test]
    fn test_rt_offset_regression() {
        let spec = "output a @10Hz := a.offset(by: -100ms).defaults(to: 0) + 1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_regression2() {
        let spec = "
            output x @ 10Hz := 1
            output x_diff := x - x.offset(by:-1s).defaults(to: x)
        ";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-1s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }
    #[test]
    fn test_rt_offset_skip2() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-2s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_fail() {
        let spec = "output a: Int8 @0.5Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        let tb = check_expect_error(spec);
        assert_eq!(1, tb.handler.emitted_errors());
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_casting_implicit_types() {
        let spec = "input x: UInt8\noutput y: Float32 := cast(x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_casting_explicit_types() {
        let spec = "input x: Int32\noutput y: UInt32 := cast<Int32,UInt32>(x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_missing_expression() {
        // should not produce an error as we want to be able to handle incomplete specs in analysis
        let spec = "input x: Bool\noutput y: Bool := \ntrigger (y || x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn infinite_recursion_regression() {
        // this should fail in type checking as the value type of `c` cannot be determined.
        let spec = "output c := c.defaults(to:0)";
        assert_eq!(1, num_type_errors(spec));
    }
    */
    /*
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
}
