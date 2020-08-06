use super::*;
extern crate regex;

use crate::pacing_types::{parse_abstract_type, AbstractPacingType, Freq};
use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Constant, Expression, Input, Output, Trigger};
use front::ast::{ExpressionKind, RTLolaAst};
use front::parse::NodeId;
use rusttyc::{TcErr, TcKey, TypeChecker};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable(String);

impl rusttyc::TcVar for Variable {}

pub struct Context<'a> {
    pub(crate) tyc: TypeChecker<AbstractPacingType, Variable>,
    pub(crate) decl: &'a DeclarationTable,
    pub(crate) node_key: HashMap<NodeId, TcKey>,
    pub(crate) bdd_vars: BddVariableSet,
}

impl<'a> Context<'a> {
    pub fn new(ast: &RTLolaAst, decl: &'a DeclarationTable) -> Context<'a> {
        let mut bdd_var_builder = BddVariableSetBuilder::new();
        let mut node_key = HashMap::new();
        let mut tyc = TypeChecker::new();

        for input in &ast.inputs {
            bdd_var_builder.make_variable(input.name.name.as_str());
            let key = tyc.get_var_key(&Variable(input.name.name.clone()));
            node_key.insert(input.id, key);
        }
        for output in &ast.outputs {
            bdd_var_builder.make_variable(output.name.name.as_str());
            let key = tyc.get_var_key(&Variable(output.name.name.clone()));
            node_key.insert(output.id, key);
        }
        for ast_const in &ast.constants {
            bdd_var_builder.make_variable(ast_const.name.name.as_str());
            let key = tyc.get_var_key(&Variable(ast_const.name.name.clone()));
            node_key.insert(ast_const.id, key);
        }
        let bdd_vars = bdd_var_builder.build();

        Context {
            tyc,
            decl,
            node_key,
            bdd_vars,
        }
    }

    pub fn input_infer(&mut self, input: &Input) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Event(self.bdd_vars.mk_var_by_name(input.name.name.as_str()));
        let input_key = self.node_key[&input.id];
        self.tyc.impose(input_key.concretizes_explicit(ac))
    }

    pub fn constant_infer(&mut self, constant: &Constant) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Event(self.bdd_vars.mk_true());
        let const_key = self.node_key[&constant.id];
        self.tyc.impose(const_key.concretizes_explicit(ac))
    }

    pub fn trigger_infer(&mut self, trigger: &Trigger) -> Result<(), TcErr<AbstractPacingType>> {
        let ex_key = self.expression_infer(&trigger.expression)?;
        let trigger_key = self.tyc.new_term_key();
        self.node_key.insert(trigger.id, trigger_key);
        self.tyc.impose(trigger_key.concretizes(ex_key))
    }

    pub fn output_infer(&mut self, output: &Output) -> Result<(), TcErr<AbstractPacingType>> {
        // Type declared AC
        let declared_ac = match parse_abstract_type(output.extend.expr.as_ref(), &self.bdd_vars) {
            Ok(b) => b,
            Err((reason, span)) => {
                // Todo: Handle properly once the interface is done
                unimplemented!();
            }
        };
        let declared_ac_key = self.tyc.new_term_key();
        self.tyc
            .impose(declared_ac_key.concretizes_explicit(declared_ac))?;
        self.node_key.insert(output.extend.id, declared_ac_key);

        // Type Expression
        let exp_key = self.expression_infer(&output.expression)?;

        // Infer resulting type
        let output_key = self.node_key[&output.id];
        self.tyc.impose(output_key.is_meet_of(declared_ac_key, exp_key))
    }

    pub fn expression_infer(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey, TcErr<AbstractPacingType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::Lit(_) => {
                let literal_type = Event(self.bdd_vars.mk_true());
                self.tyc.impose(term_key.concretizes_explicit(literal_type))?;
            }
            ExpressionKind::Ident(_) => {
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
                let key = self.node_key[&node_id];
                self.tyc.impose(term_key.equate_with(key))?;
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infer(ex)?;
                match kind {
                    Sync => self.tyc.impose(term_key.equate_with(ex_key))?,
                    Optional | Hold => self.tyc.impose(term_key.concretizes_explicit(Any))?,
                };
            }
            ExpressionKind::Default(ex, default) => {
                self.expression_infer(&*ex)?; //Option<X>
                let def_key = self.expression_infer(&*default)?; // Y

                // Is this correct?
                self.tyc.impose(term_key.equate_with(def_key))?;
            }
            ExpressionKind::Offset(expr, _) => {
                let ex_key = self.expression_infer(&*expr)?; // X

                self.tyc.impose(term_key.equate_with(ex_key))?;
            }
            ExpressionKind::SlidingWindowAggregation { .. } => {
                self.tyc
                    .impose(term_key.concretizes_explicit(AbstractPacingType::Periodic(Freq::Any)))?;
                // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
            }
            ExpressionKind::Binary(_, left, right) => {
                let left_key = self.expression_infer(&*left)?; // X
                let right_key = self.expression_infer(&*right)?; // X

                self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
            }
            ExpressionKind::Unary(_, expr) => {
                let ex_key = self.expression_infer(&*expr)?; // expr

                self.tyc.impose(term_key.equate_with(ex_key))?;
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let cond_key = self.expression_infer(&*cond)?;
                let cons_key = self.expression_infer(&*cons)?;
                let alt_key = self.expression_infer(&*alt)?;

                self.tyc.impose(term_key.is_meet_of_all(&vec![cond_key, cons_key, alt_key]))?;
            }
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(elements) => {
                let ele_keys: Vec<TcKey> = elements.iter().map(|e| self.expression_infer(&*e)).collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>()?;
                self.tyc.impose(term_key.is_meet_of_all(&ele_keys))?;
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(body, _, _, args) => {
                let body_key = self.expression_infer(&*body)?;
                let mut arg_keys: Vec<TcKey> = args.iter().map(|e| self.expression_infer(&*e)).collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>()?;
                arg_keys.push(body_key);
                self.tyc.impose(term_key.is_meet_of_all(&arg_keys))?;
            }
            ExpressionKind::Function(_, _, args) => {
                // check for name in context
                let decl = self
                    .decl
                    .get(&exp.id)
                    .expect("declaration checked by naming analysis")
                    .clone();

                match decl {
                    Declaration::ParamOut(out) => {
                        let output_key = self.node_key[&out.id];
                        self.tyc.impose(term_key.concretizes(output_key))?;
                    }
                    _ => {}
                };
                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.tyc.impose(term_key.concretizes(arg_key))?;
                }
            }
            ExpressionKind::ParenthesizedExpression(_, _, _) => unimplemented!(),
        };
        self.node_key.insert(exp.id, term_key);
        Ok(term_key)
        //Err(String::from("Error"))
    }
}
