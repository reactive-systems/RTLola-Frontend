use super::*;
extern crate regex;

use crate::pacing_types::{AbstractPacingType, Freq};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Expression};
use front::ast::{LolaSpec, ExpressionKind};
use front::parse::NodeId;
use rusttyc::{TcKey, TypeChecker};
use std::collections::HashMap;
use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable(String);

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
        let mut node_key = HashMap::new();
        let mut tyc = TypeChecker::new();

        for input in &ast.inputs{
            bdd_var_builder.make_variable(input.name.name.as_str());
            let key = tyc.get_var_key(&Variable(input.name.name.clone()));
            node_key.insert(input.id, key);
        }
        for output in &ast.outputs{
            bdd_var_builder.make_variable(output.name.name.as_str());
            let key = tyc.get_var_key(&Variable(output.name.name.clone()));
            node_key.insert(output.id, key);
        }
        for ast_const in &ast.constants{
            bdd_var_builder.make_variable(ast_const.name.name.as_str());
            let key = tyc.get_var_key(&Variable(ast_const.name.name.clone()));
            node_key.insert(ast_const.id, key);
        }
        let bdd_vars = bdd_var_builder.build();

        Context{
            tyc,
            decl,
            node_key,
            bdd_vars,
        }
    }

    pub fn expression_infer(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey<AbstractPacingType>, <AbstractPacingType as rusttyc::Abstract>::Err> {
        let term_key: TcKey<AbstractPacingType> = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::Lit(lit) => {
                let literal_type = Event(self.bdd_vars.mk_true());
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
                let ex_key = self.expression_infer(ex)?;
                match kind {
                    Sync => self.tyc.impose(term_key.unify_with(ex_key))?,
                    Optional | Hold => self.tyc.impose(term_key.captures(Any))?,
                };
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infer(&*ex)?; //Option<X>
                let def_key = self.expression_infer(&*default)?; // Y

                self.tyc.impose(term_key.unify_with(ex_key));
                self.tyc.impose(term_key.unify_with(def_key));
            }
            ExpressionKind::Offset(expr, offset) => {
                let ex_key = self.expression_infer(&*expr)?; // X

                self.tyc.impose(term_key.unify_with(ex_key));

            }
            ExpressionKind::SlidingWindowAggregation {
                expr,
                duration,
                wait,
                aggregation: aggr,
            } => {
                self.tyc.impose(term_key.captures(AbstractPacingType::Periodic(Freq::Any)));
                // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
            }
            ExpressionKind::Binary(op, left, right) => {
                let left_key = self.expression_infer(&*left)?; // X
                let right_key = self.expression_infer(&*right)?; // X

                self.tyc.impose(term_key.unify_with(left_key));
                self.tyc.impose(term_key.unify_with(right_key));
            }
            ExpressionKind::Unary(op, expr) => {
                let ex_key = self.expression_infer(&*expr)?; // expr

                self.tyc.impose(term_key.unify_with(ex_key));
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let cond_key = self.expression_infer(&*cond)?;
                let cons_key = self.expression_infer(&*cons)?;
                let alt_key = self.expression_infer(&*alt)?;

                self.tyc.impose(term_key.unify_with(cond_key));
                self.tyc.impose(term_key.unify_with(cons_key));
                self.tyc.impose(term_key.unify_with(alt_key));
            }
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(elements) => {
                for element in elements{
                    let element_key = self.expression_infer(&*element)?;
                    self.tyc.impose(term_key.unify_with(element_key));
                }
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(body, _, _, args) => {
                let body_key = self.expression_infer(&*body)?;
                self.tyc.impose(term_key.unify_with(body_key));

                for arg in args{
                    let arg_key = self.expression_infer(&*arg)?;
                    self.tyc.impose(term_key.unify_with(arg_key));
                }
            },
            ExpressionKind::Function(name, types, args) => {
                // check for name in context
                let decl = self
                    .decl
                    .get(&exp.id)
                    .expect("declaration checked by naming analysis")
                    .clone();

                match decl {
                    Declaration::ParamOut(out) => {
                        let output_key = self.node_key[&out.id];
                        self.tyc.impose(term_key.unify_with(output_key))

                    }
                    _ => {},
                };
                for arg in args{
                    let arg_key = self.expression_infer(&*arg)?;
                    self.tyc.impose(term_key.unify_with(arg_key));
                }
            }
            ExpressionKind::ParenthesizedExpression(_, _, _) => unimplemented!(),
            ExpressionKind::Aggregation(_, _, _) => unimplemented!(),
        };
        self.node_key.insert(exp.id, term_key);
        Ok(term_key)
        //Err(String::from("Error"))
    }
}