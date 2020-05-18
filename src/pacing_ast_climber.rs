use super::*;
extern crate regex;

use crate::pacing_types_bdd::{AbstractPacingType};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Expression, LitKind, Literal, Type};
use front::ast::{LolaSpec, ExpressionKind, FunctionName, Output, Parameter, TypeKind};
use front::parse::NodeId;
use front::ty::{TypeConstraint, ValueTy};
use rusttyc::{Abstract, TcKey, TypeChecker};
use std::collections::HashMap;
use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Variable {
    pub name: String,
}

impl rusttyc::TcVar for Variable {}

pub struct Context<'a> {
    pub(crate) tyc: TypeChecker<AbstractPacingType, Variable>,
    pub(crate) decl: DeclarationTable<'a>,
    pub(crate) node_key: HashMap<NodeId, TcKey<AbstractPacingType>>,
    pub(crate) bdd_vars: BddVariableSet,
}

impl<'a> Context<'a> {
    pub fn new(ast: &LolaSpec, decl: DeclarationTable<'a>) -> Context<'a>{
        let mut bdd_var_builder = BddVariableSetBuilder::new();
        for input in ast.inputs{
            bdd_var_builder.make_variable(input.name.name.as_str());
        }
        for output in ast.outputs{
            bdd_var_builder.make_variable(output.name.name.as_str());
        }
        for ast_const in ast.constants{
            bdd_var_builder.make_variable(ast_const.name.name.as_str());
        }
        let bdd_vars = bdd_var_builder.build();
        let tyc = TypeChecker::new();
        let node_key = HashMap::new();

        Context{
            tyc,
            decl,
            node_key,
            bdd_vars,
        }
    }

    pub fn expression_infere(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey<AbstractPacingType>, <AbstractPacingType as rusttyc::Abstract>::Error> {
        let term_key: TcKey<AbstractPacingType> = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = Event(self.bdd_vars.mk_true());
                self.tyc.impose(term_key.captures(literal_type));
            }
            ExpressionKind::Ident(id) => {
                //let decl = self.decl[&exp.id];
                return Ok(self.tyc.get_var_key(&Variable {
                    name: id.name.clone(),
                }));
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infere(ex)?;
                let ty = match kind {
                    Sync => self.tyc.impose(term_key.unify_with(ex_key)),
                    Optional | Hold => self.tyc.impose(term_key.captures(Any)),
                };

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
                    WindowOperation::Min | WindowOperation::Max => {}
                    //Count: (Num|Bool) -> uint
                    WindowOperation::Count => {
                        //self.tyc.fork(); //TODO
                        //check for bool case
                        self.tyc.impose(ex_key.captures(IAbstractType::Bool));
                        //OR
                        //self.tyc.impose(ex_key.captures(IAbstractType::Numeric));

                        self.tyc
                            .impose(term_key.captures(IAbstractType::UInteger(1)));
                    }
                    //all others :<T:Num>  -> T
                    _ => {
                        self.tyc.impose(ex_key.captures(IAbstractType::Numeric));
                        self.tyc.impose(term_key.unify_with(ex_key));
                    }
                }
            }
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infere(&*left)?; // X
                let right_key = self.expression_infere(&*right)?; // X

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
                    // Num x NUm -> Bool COMPARATORS
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
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
                let cond_key = self.expression_infere(&*cond)?; // Bool
                let cons_key = self.expression_infere(&*cons)?; // X
                let alt_key = self.expression_infere(&*alt)?; // X

                //Bool x T x T -> T
                self.tyc.impose(cond_key.captures(IAbstractType::Bool));

                self.tyc.impose(term_key.unify_with(cons_key));
                self.tyc.impose(term_key.unify_with(alt_key));
                //TODO Backpropagation now internal ?
            }
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(vec) => {
                unimplemented!() //TODO
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
            ExpressionKind::Function(name, types, args) => {
                //transform Type into new internal types.
                let types_vec: Vec<IAbstractType> = types
                    .iter()
                    .map(|t| self.type_kind_match(&t))
                    .collect();
                // check for name in context
                let decl = self
                    .decl
                    .get(&exp.id)
                    .expect("declaration checked by naming analysis")
                    .clone();
                match decl {
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

                        for (arg, param) in args.iter().zip(fun_decl.parameters.iter()) {
                            let p = self.replace_type(param, &generics);
                            let arg_key = self.expression_infere(&*arg)?;
                            self.tyc.impose(arg_key.unify_with(p));
                        }

                        let return_type = self.replace_type(&fun_decl.return_type, &generics);

                        self.tyc.impose(term_key.unify_with(return_type));
                    }
                    Declaration::ParamOut(out) => {
                        let params: &[Parameter] = out.params.as_slice();

                        let param_types: Vec<IAbstractType> = params
                            .iter()
                            .map(|p| self.type_kind_match(&p.ty))
                            .collect();

                        for (arg, param_t) in args.iter().zip(param_types.iter()) {
                            let arg_key = self.expression_infere(&*arg)?;
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
}