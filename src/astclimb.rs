use super::*;
extern crate regex;

use crate::types::{IAbstractType, RecursiveType};
use front::ast::{ExpressionKind, TypeKind, FunctionName, Output};
use front::ast::{Expression, LitKind, Literal};
use front::analysis::naming::{DeclarationTable, Declaration};
use rusttyc::{Abstract, TcKey, TypeChecker};
use front::ty::{ValueTy, TypeConstraint};

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct Context<'a> {
    pub(crate) tyc: TypeChecker<IAbstractType, Variable>,
    pub decl: DeclarationTable<'a>,
}

impl <'a> Context<'a> {
    pub fn expression_infere(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey<IAbstractType>, <IAbstractType as rusttyc::Abstract>::Error> {
        let term_key: TcKey<IAbstractType> = self.tyc.new_term_key();
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = match &lit.kind {
                    LitKind::Str(_) | LitKind::RawStr(_) => IAbstractType::TString,
                    LitKind::Numeric(n, post) => get_abstract_type_of_string_value(&n)?,
                    LitKind::Bool(_) => IAbstractType::Bool,
                };
                self.tyc.impose(term_key.captures(literal_type));
            }
            ExpressionKind::Ident(id) => {
                return Ok(self.tyc.get_var_key(&Variable {
                    name: id.name.clone(),
                }));
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infere(&*ex)?;
                let ty = IAbstractType::Any; //TODO FIXME
                //match kind {
                    //Sync => IAbstractType::Option(Box::new(self.tyc.get_type(ex_key))), //TODO change when no longer everything is optional
                    //Optional | Hold => IAbstractType::Option(Box::new(self.tyc.get_type(ex_key))),
                //};
                self.tyc.impose(term_key.captures(ty));
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infere(&*ex)?; //Option<X>
                let def_key = self.expression_infere(&*default)?; // Y

                let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                self.tyc.impose(m_key.key().unify_with(ex_key));
                self.tyc.impose(m_key.child().unify_with(def_key));

                // meet(X,Y)
                let result_constaint = term_key.unify_with(def_key); //TODO review
                self.tyc.impose(result_constaint);
            }
            ExpressionKind::Offset(expr, offset) => {
                let ex_key = self.expression_infere(&*expr)?; // X
                //Want build: Option<X>


                //TODO check for different result per offset
                let m_key = self.tyc.new_monad_key(RecursiveType::Option);
                self.tyc.impose(m_key.child().unify_with(ex_key));
                //m_key.key().captures( t -> Option(t));
                self.tyc.impose(term_key.unify_with(m_key.key()));

                //self.tyc.impose(term_key.bound_by_abstract(/*Option of expr type*/)); //TODO
            }
            ExpressionKind::SlidingWindowAggregation {
                expr,
                duration,
                wait,
                aggregation: aggr,
            } => {
                let ex_key = self.expression_infere(&*expr)?;
                let duration_key = self.expression_infere(&*duration)?;

                //TODO handle different wait and aggregation formulas
                self.tyc
                    .impose(duration_key.captures(IAbstractType::Numeric));

                use front::ast::WindowOperation;
                match aggr {
                    //Min|Max <T:Num> T -> Option<T>
                    WindowOperation::Min | WindowOperation::Max => {

                    }
                    //Count: (Num|Bool) -> uint
                    WindowOperation::Count => {
                        //self.tyc.fork(); //TODO
                        //check for bool case
                        self.tyc.impose(ex_key.captures(IAbstractType::Bool));
                        //OR
                        //self.tyc.impose(ex_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.captures(IAbstractType::UInteger(1)));
                    }
                    //all others :<T:Num>  -> T
                    _ => {
                        self.tyc
                            .impose(ex_key.captures(IAbstractType::Numeric));
                        self.tyc
                            .impose(term_key.unify_with(ex_key));
                    }
                }
            }
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infere(&*left)?; // X
                let right_key = self.expression_infere(&*right)?; // X

                use front::ast::BinOp;
                match op {
                    // Num x Num -> Num
                    BinOp::Add |
                    BinOp::Sub |
                    BinOp::Mul |
                    BinOp::Div |
                    BinOp::Rem |
                    BinOp::Pow => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(left_key));
                        self.tyc.impose(term_key.unify_with(right_key));
                    }
                    // Bool x Bool -> Bool
                    BinOp::And |
                    BinOp::Or => {
                        self.tyc.impose(left_key.captures(IAbstractType::Bool));
                        self.tyc.impose(right_key.captures(IAbstractType::Bool));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    }
                    // Num x Num -> Num
                    BinOp::BitXor |
                    BinOp::BitAnd |
                    BinOp::BitOr |
                    BinOp::Shl |
                    BinOp::Shr => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(left_key));
                        self.tyc.impose(term_key.unify_with(right_key));
                    }
                    // Num x NUm -> Bool COMPARATORS
                    BinOp::Eq |
                    BinOp::Lt |
                    BinOp::Le |
                    BinOp::Ne |
                    BinOp::Ge |
                    BinOp::Gt => {
                        self.tyc.impose(left_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(right_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    }
                }
            }
            ExpressionKind::Unary(op, expr) => {
                let ex_key = self.expression_infere(&*expr)?; // expr

                use front::ast::UnOp;
                match op {
                    //Num -> Num
                    UnOp::BitNot |
                    UnOp::Neg => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Numeric));

                        self.tyc.impose(term_key.unify_with(ex_key));
                    },
                    // Bool -> Bool
                    UnOp::Not => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Bool));

                        self.tyc.impose(term_key.captures(IAbstractType::Bool));
                    },
                }

            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let cond_key = self.expression_infere(&*cond)?; // Bool
                let cons_key = self.expression_infere(&*cons)?; // X
                let alt_key = self.expression_infere(&*alt)?; // X

                //Bool x T x T -> T
                self.tyc.impose(cond_key.captures(IAbstractType::Bool));

                self.tyc.impose(term_key.unify_with(cons_key));
                self.tyc.impose(term_key.unify_with(alt_key));
                //TODO Backpropagation now internal ?
            }
            ExpressionKind::MissingExpression => {
                unreachable!()
            }
            ExpressionKind::Tuple(vec) => {
                unimplemented!() //TODO
            }
            ExpressionKind::Field(_, _) => {
                unimplemented!()
            }
            ExpressionKind::Method(_, _, _, _) => {
                unimplemented!()
            }
            ExpressionKind::Function(name, types, args) => {
                //transform Type into new internal types.
                let types_vec :Vec<IAbstractType> = types.iter().map(|t| type_kind_match(&t.kind)).collect();
                // check for name in context
                let decl = self.decl.get(&exp.id).expect("declaration checked by naming analysis");
                match decl {
                    Declaration::Func(fun_decl) => {
                        //Generics
                        let generics: TcKey<IAbstractType> = fun_decl.generics
                            .iter()
                            .map(|gen| {
                                let gen_key: TcKey<IAbstractType> = self.tyc.new_term_key();
                                match &gen {
                                    ValueTy::Constr(tc) => {
                                        let cons = match_constraint(tc);
                                        self.tyc.impose(gen_key.captures(cons));
                                    }
                                    _ => unreachable!()
                                };
                                gen_key
                            })
                            .collect();

                        let params: Vec<IAbstractType> = fun_decl.parameters.clone().iter().map(
                            |p| {
                                replace_type(p)
                            }
                        ).collect();

                        for ((arg,param_type), type_value) in args.iter().zip(fun_decl.parameters.iter()).zip(types_vec.iter()) {
                            let arg_key = self.expression_infere(&*arg)?;
                            self.tyc.impose(arg_key.captures(type_value.clone()));

                            self.tyc.impose(arg_key.captures(value_type_match(param_type)));


                            //let declared_param_type = fun_decl.
                        }
                    },
                    Declaration::ParamOut(out) => {
                        unimplemented!()
                    }
                    _ => unreachable!("ensured by naming analysis"),
                };
                // create matching function declaration
                //type check each argument
                /*for (arg,t) in args.iter().zip(types.iter()) {
                    let sub_ex_key = self.expression_infere(&*arg)?;

                    self.tyc.impose(sub_ex_key.captures(typekind_match(&t.kind.clone())));

                }*/
            }
            ExpressionKind::ParenthesizedExpression(_, _, _) => unimplemented!(),
            ExpressionKind::Aggregation(_, _, _) => unimplemented!(),
        };
        Ok(term_key)
        //Err(String::from("Error"))
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
    let pat = regex::Regex::new("\"*\"").unwrap();
    if pat.is_match(value_str) {
        return Ok(IAbstractType::TString);
    }
    Err(String::from(format!(
        "Non matching String Literal: {}",
        value_str
    )))
}

fn type_kind_match(kind: &TypeKind) -> IAbstractType {
    //TODO
    unimplemented!()
}
fn value_type_match(vt: &ValueTy) -> IAbstractType {
    //TODO
    unimplemented!()
}

fn match_constraint(cons: &TypeConstraint) -> IAbstractType {
    //TODO
    unimplemented!()
}

fn replace_type(vt: &ValueTy) -> IAbstractType {
    //TODO
    unimplemented!()
}