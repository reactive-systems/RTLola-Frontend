use super::*;
extern crate regex;

use crate::value_types::IAbstractType;
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Constant, Input, Output, Trigger};
use front::ast::{Expression, LitKind, Type};
use front::ast::{ExpressionKind, Parameter, TypeKind};
use front::parse::NodeId;
use front::ty::{TypeConstraint, ValueTy};
use rusttyc::{TcKey, TypeChecker, TcErr};
use rusttyc::types::{Abstract};
use std::collections::HashMap;
use std::rc::Rc;
use bimap::BiMap;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct ValueContext {
    pub tyc: TypeChecker<IAbstractType, Variable>,
    pub decl: DeclarationTable,
    pub node_key: BiMap<NodeId, TcKey>, //TODO initialisiere tabelle
}

impl ValueContext {
    pub fn new(ast: &RTLolaAst, decl: DeclarationTable) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = BiMap::new();

        for input in &ast.inputs {
            node_key.insert(
                input.id,
                tyc.get_var_key(&Variable {
                    name: input.name.name.clone(),
                }),
            );
        }

        for cons in &ast.constants {
            node_key.insert(
                cons.id,
                tyc.get_var_key(&Variable {
                    name: cons.name.name.clone(),
                }),
            );
        }

        for out in &ast.outputs {
            node_key.insert(
                out.id,
                tyc.get_var_key(&Variable {
                    name: out.name.name.clone(),
                }),
            );
        }

        ValueContext {
            tyc,
            decl,
            node_key,
        }
    }

    pub fn input_infer(&mut self, input: &Input) -> Result<TcKey, TcErr<IAbstractType>>{
        let term_key: TcKey = *self.node_key.get_by_left(&input.id).expect("");
        //Annotated Type
        if let t= &input.ty {
            let annotated_type_replaced = self.type_kind_match(t);
            self.tyc
                .impose(term_key.concretizes_explicit(annotated_type_replaced))?;
        };
        let mut param_types = Vec::new();
        for param in &input.params {
            let param_key = self.tyc.get_var_key(&Variable{name: param.name.name.clone()});
            let t = self.type_kind_match(&param.ty);
            self.tyc.impose(param_key.concretizes_explicit(t))?;
            param_types.push(param_key);
        }
        assert!(param_types.is_empty(),"parametric input types currently not supported");
        Ok(term_key)

    }

    pub fn constant_infer(
        &mut self,
        cons: &Constant,
    ) -> Result<TcKey, TcErr<IAbstractType>>  {
        let term_key: TcKey = *self.node_key.get_by_left(&cons.id).expect("");
        //Annotated Type
        if let Some(t) = &cons.ty {
            let annotated_type_replaced = self.type_kind_match(t);
            self.tyc
                .impose(term_key.concretizes_explicit(annotated_type_replaced))?;
        }
        //Type from Literal
        let lit_type = self.match_lit_kind(cons.literal.kind.clone());
        self.tyc.impose(term_key.concretizes_explicit(lit_type))?;

        self.node_key.insert(cons.id, term_key);
        return Ok(term_key);
    }

    pub fn output_infer(&mut self, out: &Output) -> Result<TcKey,TcErr<IAbstractType>> {
        let out_key = *self.node_key.get_by_left(&out.id).expect("Added in constructor");

        if let t= &out.ty {
            let annotated_type_replaced = self.type_kind_match(t);
            self.tyc
                .impose(out_key.concretizes_explicit(annotated_type_replaced))?;
        };
        let mut param_types = Vec::new();
        for param in &out.params {
            let param_key = self.tyc.get_var_key(&Variable{ name: param.name.name.clone()});
            let t = self.type_kind_match(&param.ty);
            self.tyc.impose(param_key.concretizes_explicit(t))?;
            param_types.push(param_key);
        }

        let expression_key = self.expression_infer(&out.expression,None)?;
        self.tyc.impose(out_key.concretizes(expression_key))?;
        Ok(out_key)
    }

    pub fn trigger_infer(&mut self, tr: &Trigger) -> Result<TcKey, TcErr<IAbstractType>> {
        let tr_key = *self.node_key.get_by_left(&tr.id).expect("Added in constructor");
        let expression_key = self.expression_infer(&tr.expression,Some(IAbstractType::Bool))?;
        self.tyc.impose(tr_key.concretizes(expression_key))?;
        Ok(tr_key)
    }

    fn expression_infer(
        &mut self,
        exp: &Expression,
        target_type: Option<IAbstractType>,
    ) -> Result<TcKey, TcErr<IAbstractType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        if let Some(t) = target_type {
            self.tyc.impose(term_key.concretizes_explicit(t))?;
        }
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = self.match_lit_kind(lit.kind.clone());
                self.tyc.impose(term_key.concretizes_explicit(literal_type))?;
            }
            ExpressionKind::Ident(id) => {
                let decl = &self.decl[&exp.id];
                let node_id = match decl {
                    Declaration::Const(c) => c.id,
                    Declaration::Out(out) => out.id,
                    Declaration::ParamOut(param) => param.id,
                    Declaration::In(input) => input.id,
                    Declaration::Type(_) | Declaration::Param(_) | Declaration::Func(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                };
                let key = self.node_key.get_by_left(&node_id).expect("Value should be contained");
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
                    self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                    let inner_key = self.tyc.get_child_key(term_key,1) ?;
                    self.tyc.impose(ex_key.equate_with(inner_key))?;
                    }
                };
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infer(&*ex, None)?; //Option<X>
                let def_key = self.expression_infer(&*default, None)?; // Y


                self.tyc.impose( ex_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                let inner_key = self.tyc.get_child_key(ex_key,1) ?;
                self.tyc.impose(def_key.equate_with(inner_key))?;
                self.tyc.impose(term_key.equate_with(def_key))?;
            }
            ExpressionKind::Offset(expr, offset) => {
                let ex_key = self.expression_infer(&*expr, None)?; // X
                                                                   //Want build: Option<X>

                //TODO check for different offset - there are no realtime offsets so far
                self.tyc.impose(term_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                let inner_key = self.tyc.get_child_key(term_key,1) ?;
                self.tyc.impose(ex_key.equate_with(inner_key))?;

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
                        self.tyc.impose( term_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                        let inner_key = self.tyc.get_child_key(term_key,1) ?;
                        self.tyc.impose(inner_key.equate_with(ex_key))?;
                    }
                    //Count: Any -> uint
                    WindowOperation::Count => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Any))?;
                        self.tyc
                            .impose(term_key.concretizes_explicit(IAbstractType::UInteger(1)))?;
                    }
                    //all others :<T:Num>  -> T
                    WindowOperation::Integral => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Float(1)))?; //TODO maybe numeric
                        if *wait {
                            self.tyc.impose( term_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                            let inner_key = self.tyc.get_child_key(term_key,1) ?;
                            self.tyc.impose(inner_key.equate_with(ex_key))?;
                        } else {
                            self.tyc.impose(term_key.concretizes(ex_key))?;
                        }
                    }
                    WindowOperation::Sum | WindowOperation::Product => {
                        self.tyc
                            .impose(ex_key.concretizes_explicit(IAbstractType::Numeric))?;
                        if *wait {
                            self.tyc.impose( term_key.concretizes_explicit(IAbstractType::Option(IAbstractType::Any.into())))?;
                            let inner_key = self.tyc.get_child_key(term_key,1) ?;
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
                    } //TODO
                }
            }
            //TODO
            ///// implicit widening requieres join operand
            // a + b -> c c = meet(a,b) then equate a and b with join(a,b) //FIXME
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infer(&*left, None)?; // X
                let right_key = self.expression_infer(&*right, None)?; // X

                use front::ast::BinOp;
                match op {
                    // Num x Num -> Num
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                        self.tyc
                            .impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc
                            .impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;

                        self.tyc.impose(term_key.is_meet_of(left_key,right_key))?;
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


                        self.tyc.impose(term_key.is_meet_of(left_key,right_key))?;
                        self.tyc.impose(term_key.equate_with(left_key))?;
                        self.tyc.impose(term_key.equate_with(right_key))?;
                    }
                    // Num x Num -> Bool COMPARATORS
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                        self.tyc
                            .impose(left_key.concretizes_explicit(IAbstractType::Numeric))?;
                        self.tyc
                            .impose(right_key.concretizes_explicit(IAbstractType::Numeric))?;
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
                let cond_key = self.expression_infer(&*cond, Some(IAbstractType::Bool))?; // Bool
                let cons_key = self.expression_infer(&*cons, None)?; // X
                let alt_key = self.expression_infer(&*alt, None)?; // X

                //Bool x T x T -> T
                //self.tyc.impose(cond_key.captures(IAbstractType::Bool)); //TODO check me if this is right

                self.tyc.impose(term_key.is_sym_meet_of(cons_key,alt_key))?;
                self.tyc.impose(cons_key.equate_with(alt_key))?;
            }
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(vec) => {
                unimplemented!() //TODO
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
            ExpressionKind::Function(name, types, args) => {
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
                        //Generics
                        let generics: Vec<TcKey> = fun_decl
                            .generics
                            .iter()
                            .map(|gen| {
                                let gen_key: TcKey = self.tyc.new_term_key();
                                let rusttyc_result = match &gen {
                                    ValueTy::Constr(tc) => {
                                        let cons = match_constraint(tc);
                                        self.tyc.impose(gen_key.concretizes_explicit(cons))
                                    }
                                    _ => unreachable!(),
                                };
                                rusttyc_result.map(|_ | gen_key)
                            })
                            .collect::<Result<Vec<TcKey>,TcErr<IAbstractType>>>()?;

                        for (t, gen) in types_vec.iter().zip(generics.iter()) {
                            let t_key = self.tyc.new_term_key();
                            self.tyc.impose(t_key.concretizes_explicit(t.clone()))?;
                            self.tyc.impose(t_key.concretizes(*gen))?;
                        }
                        //FOR: type.captures(generic)

                        for (arg, param) in args.iter().zip(fun_decl.parameters.iter()) {
                            let p = self.replace_type(param, &generics)?;
                            let arg_key = self.expression_infer(&*arg, None)?;
                            self.tyc.impose(arg_key.concretizes(p))?;
                        }

                        let return_type = self.replace_type(&fun_decl.return_type, &generics)?;

                        self.tyc.impose(term_key.concretizes(return_type))?;
                    }
                    Declaration::ParamOut(out) => {
                        let params: &[Rc<Parameter>] = out.params.as_slice();

                        let param_types: Vec<IAbstractType> =
                            params.iter().map(|p| self.type_kind_match(&p.ty)).collect();

                        for (arg, param_t) in args.iter().zip(param_types.iter()) {
                            let arg_key = self.expression_infer(&*arg, None)?;
                            self.tyc.impose(arg_key.concretizes_explicit(param_t.clone()))?;
                        }

                        self.tyc
                            .impose(term_key.concretizes_explicit(self.type_kind_match(&out.ty)))?;
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

    fn replace_type(&mut self, vt: &ValueTy, to: &[TcKey]) -> Result<TcKey,TcErr<IAbstractType>> {
        match vt {
            &ValueTy::Param(idx, _) => Ok(to[idx as usize]),
            ValueTy::Option(o) => {
                //TODO Option
                unimplemented!()
            }
            ValueTy::Constr(c) => {
                let key = self.tyc.new_term_key();
                self.tyc.impose(key.concretizes_explicit(match_constraint(c))).map(|_| key)
            }
            _ if vt.is_primitive() => {
                let key = self.tyc.new_term_key();
                self.tyc
                    .impose(key.concretizes_explicit(self.value_type_match(vt))).map(|_| key)
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
            TypeKind::Inferred => {
                //TODO
                unimplemented!()
            }
        }
    }

    fn value_type_match(&self, vt: &ValueTy) -> IAbstractType {
        match vt {
            ValueTy::Bool => IAbstractType::Bool,
            ValueTy::Int(i) => {
                use front::ty::IntTy;
                match i {
                    IntTy::I8 => IAbstractType::Integer(8),
                    IntTy::I16 => IAbstractType::Integer(16),
                    IntTy::I32 => IAbstractType::Integer(32),
                    IntTy::I64 => IAbstractType::Integer(64),
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
            ValueTy::Tuple(vec) => unimplemented!("TODO"),
            ValueTy::Option(o) => IAbstractType::Option(self.value_type_match(&**o).into()),
            ValueTy::Infer(_) => unreachable!(),
            ValueTy::Constr(c) => {
                use front::ty::TypeConstraint::*;
                match c {
                    //TODO check equatable and comparable
                    Integer => IAbstractType::Integer(1),
                    SignedInteger => IAbstractType::Integer(1),
                    UnsignedInteger => IAbstractType::UInteger(1),
                    FloatingPoint => IAbstractType::Float(1),
                    Numeric => IAbstractType::Numeric,
                    Equatable => IAbstractType::Any,
                    Comparable => IAbstractType::Numeric,
                    Unconstrained => IAbstractType::Any,
                }
            }
            ValueTy::Param(_, _) => {
                unimplemented!("Param case should only be addressed in replace_type(...)")
            }
            ValueTy::Error => unreachable!("Error should be checked before hand"),
        }
    }

    fn match_lit_kind(&self, lit: LitKind) -> IAbstractType {
        match lit {
            LitKind::Str(_) | LitKind::RawStr(_) => IAbstractType::TString,
            LitKind::Numeric(n, post) => get_abstract_type_of_string_value(&n).expect(""),
            LitKind::Bool(_) => IAbstractType::Bool,
        }
    }

    fn handle_error(&self, exp: &Expression, err: TcErr<IAbstractType>) -> <IAbstractType as Abstract>::Err{
        let msg = match err {
            TcErr::ChildAccessOutOfBound(key,ty,n) => {
                format!("Invalid child-type access by {:?}: {}-th child for {:?}",self.node_key.get_by_right(&key),n,ty)
            },
            TcErr::KeyEquation(k1, k2, msg) => {
                format!("Incompatible Type for equation of {:?} and {:?}: {}",self.node_key.get_by_right(&k1),self.node_key.get_by_right(&k2),msg)
            },
            TcErr::TypeBound(key,msg) => {
                format!("Invalid type bound enforced on {:?}: {}",self.node_key.get_by_right(&key),msg)
            }
        };
        msg
    }
}

fn get_abstract_type_of_string_value(value_str: &String) -> Result<IAbstractType, String> {
    let int_parse = value_str.parse::<i64>();
    if let Ok(n) = int_parse {
        return Ok(IAbstractType::Integer(64 - n.leading_zeros()));
    }
    let uint_parse = value_str.parse::<u64>();
    if let Ok(u) = uint_parse {
        return Ok(IAbstractType::UInteger(64 - u.leading_zeros()));
    }
    let float_parse = value_str.parse::<f64>();
    if let Ok(f) = float_parse {
        return Ok(IAbstractType::Float(64));
    }
    let pat = regex::Regex::new("\"*\"").unwrap(); //TODO simplify , check first and last character
    if pat.is_match(value_str) {
        return Ok(IAbstractType::TString);
    }
    Err(String::from(format!(
        "Non matching String Literal: {}",
        value_str
    )))
}

fn match_constraint(cons: &TypeConstraint) -> IAbstractType {
    //TODO
    match cons {
        TypeConstraint::Numeric => IAbstractType::Numeric,
        TypeConstraint::SignedInteger => IAbstractType::Integer(1),
        TypeConstraint::UnsignedInteger => IAbstractType::UInteger(1),
        TypeConstraint::FloatingPoint => IAbstractType::Float(1),
        TypeConstraint::Integer => {unimplemented!()},
        TypeConstraint::Equatable => {unimplemented!()},
        TypeConstraint::Comparable => {unimplemented!()},
        TypeConstraint::Unconstrained => IAbstractType::Any,
    };
    unimplemented!()
}
