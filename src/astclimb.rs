use super::*;
extern crate regex;

use crate::types::{IAbstractType, RecursiveType};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Expression, LitKind, Literal, Type};
use front::ast::{ExpressionKind, FunctionName, Output, Parameter, TypeKind};
use front::parse::NodeId;
use front::ty::{TypeConstraint, ValueTy};
use rusttyc::{Abstract, TcKey, TypeChecker};
use std::collections::HashMap;
use front::ast::Constant;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct Context<'a> {
    pub tyc: TypeChecker<IAbstractType, Variable>,
    pub decl: DeclarationTable<'a>,
    pub node_key: HashMap<NodeId, TcKey<IAbstractType>>, //TODO initialisiere tabelle
}

impl<'a> Context<'a> {

    pub fn new(ast: &LolaSpec, decl: DeclarationTable<'a>) -> Self {
        let mut tyc = TypeChecker::new();
        let mut node_key = HashMap::new();

        for input in ast.inputs {
            node_key.insert(input.id,tyc.get_var_key(Variable(input.name.name.clone())));
        }

        for cons in ast.constants {
            node_key.insert(cons.id,tyc.get_var_key(Variable(cons.name.name.clone())));
        }

        for out in ast.outputs {
            node_key.insert(out.id,tyc.get_var_key(Variable(out.name.name.clone())));
        }

        Context {
            tyc,
            decl,
            node_key,
        }
    }

    pub fn constant_infer(
        &mut self,
        cons: &Constant,
    ) -> Result<TcKey<IAbstractType>, <IAbstractType as rusttyc::Abstract>::Error> {
        let term_key: TcKey<IAbstractType> = self.tyc.new_term_key();
        //Annotated Type
        if let Some(t) = &cons.ty {
            let annotaded_type_replaced = self.type_kind_match(t);
            self.tyc.impose(term_key.captures(annotaded_type_replaced));
        }
        //Type from Literal
        let lit_type = self.match_lit_kind(cons.literal.kind.clone());
        self.tyc.impose(term_key.captures(lit_type));

        self.node_key.insert(cons.id,term_key);
        return Ok(term_key);
    }




    pub fn expression_infer(
        &mut self,
        exp: &Expression,
        target_type: Option<IAbstractType>,
    ) -> Result<TcKey<IAbstractType>, <IAbstractType as rusttyc::Abstract>::Error> {
        let term_key: TcKey<IAbstractType> = self.tyc.new_term_key();
        if let Some(t) = target_type {
            self.tyc.impose(term_key.captures(t));
        }
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = self.match_lit_kind(lit.kind.clone());
                self.tyc.impose(term_key.captures(literal_type));
            }
            ExpressionKind::Ident(id) => {
                let decl = self.decl[&exp.id];
                let node_id = match decl {
                    Declaration::Const(c) => c.id,
                    Declaration::Out(out) => out.id,
                    Declaration::ParamOut(param) => param.id,
                    Declaration::In(input) => input.id,
                    Declaration::Type(_) | Declaration::Param(_) | Declaration::Func(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                };
                let key = self.node_key[&node_id];
                self.tyc.impose(term_key.unify_with(key));
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infer(&*ex, None)?;
                 match kind {
                     Sync => {
                         //Sync access just returns the stream type
                         self.tyc.impose(term_key.unify_with(ex_key));
                     },
                     Optional | Hold => {
                         //Optional and Hold return Option<X> Type
                         let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                         self.tyc.impose(m_key.child().unify_with(ex_key));
                         self.tyc.impose(term_key.unify_with(m_key.key()));
                     },
                 };
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infer(&*ex, None)?; //Option<X>
                let def_key = self.expression_infer(&*default, None)?; // Y

                let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                self.tyc.impose(m_key.key().unify_with(ex_key));
                self.tyc.impose(m_key.child().unify_with(def_key));

                // meet(X,Y)
                let result_constraint = term_key.unify_with(def_key);
                self.tyc.impose(result_constraint);
            }
            ExpressionKind::Offset(expr, offset) => {
                let ex_key = self.expression_infer(&*expr, None)?; // X
                                                              //Want build: Option<X>

                //TODO check for different offset - there are no realtime offsets so far
                let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                self.tyc.impose(m_key.child().unify_with(ex_key));
                //m_key.key().captures( t -> Option(t));
                self.tyc.impose(term_key.unify_with(m_key.key()));
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
                    .impose(duration_key.captures(IAbstractType::Numeric));

                use front::ast::WindowOperation;
                match aggr {
                    //Min|Max|Avg <T:Num> T -> Option<T>
                    WindowOperation::Min | WindowOperation::Max | WindowOperation::Average => {
                        let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                        self.tyc.impose(m_key.child().unify_with(ex_key));
                        self.tyc.impose(term_key.unify_with(m_key.key()));
                    }
                    //Count: Any -> uint
                    WindowOperation::Count => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Any));
                        self.tyc
                            .impose(term_key.captures(IAbstractType::UInteger(1)));
                    }
                    //all others :<T:Num>  -> T
                    WindowOperation::Integral => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Float(1))); //TODO maybe numeric
                        if *wait {
                            let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                            self.tyc.impose(m_key.child().unify_with(ex_key));
                            self.tyc.impose(term_key.unify_with(m_key.key()));
                        } else {
                            self.tyc.impose(term_key.unify_with(ex_key));
                        }
                    }
                    WindowOperation::Sum | WindowOperation::Product => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Numeric));
                        if *wait {
                            let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                            self.tyc.impose(m_key.child().unify_with(ex_key));
                            self.tyc.impose(term_key.unify_with(m_key.key()));
                        } else {
                            self.tyc.impose(term_key.unify_with(ex_key));
                        }
                    }
                }
            }
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infer(&*left, None)?; // X
                let right_key = self.expression_infer(&*right, None)?; // X

                use front::ast::BinOp;
                match op {
                    // Num x Num -> Num
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(left_key));
                        self.tyc.impose(term_key.unify_with(right_key));
                    }
                    // Bool x Bool -> Bool
                    BinOp::And | BinOp::Or => {
                        self.tyc.impose(left_key.captures(IAbstractType::Bool));
                        self.tyc.impose(right_key.captures(IAbstractType::Bool));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    }
                    // Num x Num -> Num
                    BinOp::BitXor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(left_key));
                        self.tyc.impose(term_key.unify_with(right_key));
                    }
                    // Num x Num -> Bool COMPARATORS
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));
                        //TODO need unify left & right ?
                        self.tyc.impose(left_key.unify_with(right_key));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    }
                }
            }
            ExpressionKind::Unary(op, expr) => {
                let ex_key = self.expression_infer(&*expr, None)?; // expr

                use front::ast::UnOp;
                match op {
                    //Num -> Num
                    UnOp::BitNot | UnOp::Neg => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(ex_key));
                    }
                    // Bool -> Bool
                    UnOp::Not => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Bool));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    }
                }
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let cond_key = self.expression_infer(&*cond, Some(IAbstractType::Bool))?; // Bool
                let cons_key = self.expression_infer(&*cons, None)?; // X
                let alt_key = self.expression_infer(&*alt, None)?; // X

                //Bool x T x T -> T
                //self.tyc.impose(cond_key.captures(IAbstractType::Bool)); //TODO check me if this is right

                self.tyc.impose(term_key.unify_with(cons_key));
                self.tyc.impose(term_key.unify_with(alt_key));
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
                        let generics: Vec<TcKey<IAbstractType>> = fun_decl
                            .generics
                            .iter()
                            .map(|gen| {
                                let gen_key: TcKey<IAbstractType> = self.tyc.new_term_key();
                                match &gen {
                                    ValueTy::Constr(tc) => {
                                        let cons = match_constraint(tc);
                                        self.tyc.impose(gen_key.captures(cons));
                                    }
                                    _ => unreachable!(),
                                };
                                gen_key
                            })
                            .collect::<Vec<TcKey<IAbstractType>>>();

                        for (t,gen) in types_vec.iter().zip(generics.iter()) {
                            let t_key = self.tyc.new_term_key();
                            self.tyc.impose(t_key.captures(t.clone()));
                            self.tyc.impose(t_key.unify_with(*gen));
                        }
                        //FOR: type.captures(generic)

                        for (arg, param) in args.iter().zip(fun_decl.parameters.iter()) {
                            let p = self.replace_type(param, &generics);
                            let arg_key = self.expression_infer(&*arg, None)?;
                            self.tyc.impose(arg_key.unify_with(p));
                        }

                        let return_type = self.replace_type(&fun_decl.return_type, &generics);

                        self.tyc.impose(term_key.unify_with(return_type));
                    }
                    Declaration::ParamOut(out) => {
                        let params: &[Parameter] = out.params.as_slice();

                        let param_types: Vec<IAbstractType> =
                            params.iter().map(|p| self.type_kind_match(&p.ty)).collect();

                        for (arg, param_t) in args.iter().zip(param_types.iter()) {
                            let arg_key = self.expression_infer(&*arg, None)?;
                            self.tyc.impose(arg_key.captures(param_t.clone()));
                        }

                        self.tyc
                            .impose(term_key.captures(self.type_kind_match(&out.ty)));
                    }
                    _ => unreachable!("ensured by naming analysis"),
                };
            }
            ExpressionKind::ParenthesizedExpression(_, _, _) => unimplemented!(),
            ExpressionKind::Aggregation(_, _, _) => unimplemented!(),
        };
        self.node_key.insert(exp.id, term_key);
        Ok(term_key)
        //Err(String::from("Error"))
    }

    fn replace_type(&mut self, vt: &ValueTy, to: &[TcKey<IAbstractType>]) -> TcKey<IAbstractType> {
        match vt {
            &ValueTy::Param(idx, _) => to[idx as usize],
            ValueTy::Option(o) => {
                //TODO Option
                unimplemented!()
            }
            ValueTy::Constr(c) => {
                let key = self.tyc.new_term_key();
                self.tyc.impose(key.captures(match_constraint(c)));
                key
            }
            _ if vt.is_primitive() => {
                let key = self.tyc.new_term_key();
                self.tyc.impose(key.captures(self.value_type_match(vt)));
                key
            }
            _ => unreachable!("replace for {}",vt),
        }
    }

    fn type_kind_match(&self, t: &Type) -> IAbstractType {
        let kind = &t.kind;
        match kind {
            TypeKind::Simple(_) => {
                let decl = self.decl[&t.id].clone();
                if let Declaration::Type(ty) = decl {
                    self.value_type_match(ty)
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
        match vt{
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
            ValueTy::Tuple(vec) => {
                unimplemented!("TODO")
            }
            ValueTy::Option(o) => {
                IAbstractType::Option(self.value_type_match(&**o).into())
            }
            ValueTy::Infer(_) => unreachable!(),
            ValueTy::Constr(c) => {
                use front::ty::TypeConstraint;
                match c {
                    //TODO check equatable and comparable
                    TypeConstraint::Integer => IAbstractType::Integer(1),
                    TypeConstraint::SignedInteger => IAbstractType::Integer(1),
                    TypeConstraint::UnsignedInteger => IAbstractType::UInteger(1),
                    TypeConstraint::FloatingPoint => IAbstractType::Float(1),
                    TypeConstraint::Numeric => IAbstractType::Numeric,
                    TypeConstraint::Equatable => IAbstractType::Any,
                    TypeConstraint::Comparable => IAbstractType::Numeric,
                    TypeConstraint::Unconstrained => IAbstractType::Any,
                }
            }
            ValueTy::Param(_, _) => unimplemented!("Param case should only be addressed in replace_type(...)"),
            ValueTy::Error => unreachable!("Error should be checked before hand"),
        }
    }

    fn match_lit_kind(&self, lit:LitKind) -> IAbstractType {
        match lit {
            LitKind::Str(_) | LitKind::RawStr(_) => IAbstractType::TString,
            LitKind::Numeric(n, post) => get_abstract_type_of_string_value(&n).expect(""),
            LitKind::Bool(_) => IAbstractType::Bool,
        }
    }
}

fn get_abstract_type_of_string_value(value_str: &String) -> Result<IAbstractType, String> {
    let int_parse = value_str.parse::<i64>();
    if let Ok(n) = int_parse {
        return Ok(IAbstractType::Integer(
            64 - int_parse.ok().unwrap().leading_zeros(),
        ));
    }
    let uint_parse = value_str.parse::<u64>();
    if let Ok(n) = uint_parse {
        return Ok(IAbstractType::UInteger(
            64 - int_parse.ok().unwrap().leading_zeros(),
        ));
    }
    let float_parse = value_str.parse::<f64>();
    if let Ok(n) = uint_parse {
        return Ok(IAbstractType::Float(
            64 - int_parse.ok().unwrap().leading_zeros(),
        ));
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
    unimplemented!()
}
