use super::*;

use crate::pacing_ast_climber::Context as PacingContext;
use crate::pacing_types::{emit_error, ConcretePacingType};
use crate::value_ast_climber::ValueContext;
use crate::value_types::{IAbstractType, IConcreteType};
use front::analysis::naming::DeclarationTable;
use front::ast::RTLolaAst;
use front::parse::NodeId;
use front::reporting::{Handler, LabeledSpan};
use rusttyc::types::ReifiedTypeTable;
use std::collections::HashMap;
use std::hash::Hash;

pub struct LolaTypeChecker<'a> {
    pub(crate) ast: RTLolaAst,
    pub(crate) declarations: DeclarationTable,
    pub(crate) handler: &'a Handler,
}

impl<'a> LolaTypeChecker<'a> {
    pub fn new(spec: &RTLolaAst, declarations: DeclarationTable, handler: &'a Handler) -> Self {
        LolaTypeChecker {
            ast: spec.clone(),
            declarations: declarations.clone(),
            handler,
        }
    }

    pub fn check(&mut self) {
        //TODO imports
        self.value_type_infer();
        self.pacing_type_infer();
    }

    pub(crate) fn pacing_type_infer(&mut self) -> Option<HashMap<NodeId, ConcretePacingType>> {
        let mut ctx = PacingContext::new(&self.ast, &self.declarations);
        let input_names: HashMap<NodeId, &str> = self
            .ast
            .inputs
            .iter()
            .map(|i| (i.id, i.name.name.as_str()))
            .collect();

        for input in &self.ast.inputs {
            if let Err(e) = ctx.input_infer(input) {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
            }
        }

        for constant in &self.ast.constants {
            if let Err(e) = ctx.constant_infer(constant) {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
            }
        }

        for output in &self.ast.outputs {
            if let Err(e) = ctx.output_infer(output) {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
            }
        }

        for trigger in &self.ast.trigger {
            if let Err(e) = ctx.trigger_infer(trigger) {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
            }
        }

        let vars = ctx.bdd_vars.clone();
        let tt = match ctx.tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
                return None;
            }
        };

        let key_span = ctx.key_span.clone();
        let ctt: HashMap<NodeId, ConcretePacingType> = ctx
            .node_key
            .iter()
            .filter_map(|(id, key)| {
                match ConcretePacingType::from_abstract(tt[*key].clone(), &vars) {
                    Ok(ct) => Some((*id, ct)),
                    Err(e) => {
                        let ls = LabeledSpan::new(key_span[key], "Cannot infer type.", true);
                        self.handler.error_with_span(&e, ls);
                        None
                    }
                }
            })
            .collect();

        if self.handler.contains_error() {
            return None;
        }

        if let Err((reason, span)) = PacingContext::post_process(&self.ast, &ctt) {
            let ls = LabeledSpan::new(span, "here", true);
            self.handler.error_with_span(&reason, ls);
            return None;
        }

        Some(ctt)
    }

    pub(crate) fn value_type_infer(&self) -> Result<HashMap<NodeId, IConcreteType>, String> {
        //let value_tyc = rusttyc::TypeChecker::new();

        let mut ctx = ValueContext::new(&self.ast, self.declarations.clone());

        for input in &self.ast.inputs {
            if let Err(e) = ctx.input_infer(input) {
                self.handler.error_with_span(
                    "Input inference error",
                    LabeledSpan::new(input.span, "Todo", true),
                );
            }
        }

        for constant in &self.ast.constants {
            if let Err(e) = ctx.constant_infer(constant) {
                self.handler.error_with_span(
                    "Output inference error",
                    LabeledSpan::new(constant.span, "Todo", true),
                );
                return Err(ctx.handle_error(e));
            }
        }

        for output in &self.ast.outputs {
            if let Err(e) = ctx.output_infer(output) {
                self.handler.error_with_span(
                    "Output inference error",
                    LabeledSpan::new(output.span, "Todo", true),
                );
                return Err(ctx.handle_error(e));
            }
        }

        for trigger in &self.ast.trigger {
            if let Err(e) = ctx.trigger_infer(trigger) {
                self.handler.error_with_span(
                    "Output inference error",
                    LabeledSpan::new(trigger.span, "Todo", true),
                );
                return Err(ctx.handle_error(e));
            }
        }

        let tt_r = ctx.tyc.type_check();
        if let Err(tc_err) = tt_r {
            return Err("TODO".to_string());
        }
        let tt = tt_r.ok().expect("");
        let bm = ctx.node_key;
        for (nid, k) in bm.iter() {
            //DEBUGG
            println!("{:?}", (*nid, tt[*k].clone()));
        }
        let rtt_r = tt.try_reified();
        if let Err(a) = rtt_r {
            return Err("TypeTable not reifiable: ValueType not constrained enough".to_string());
        }
        let rtt: ReifiedTypeTable<IConcreteType> = rtt_r.ok().expect("");
        let mut result_map = HashMap::new();
        for (nid, k) in bm.iter() {
            result_map.insert(*nid, rtt[*k].clone());
        }
        Ok(result_map)
    }

    pub fn generate_raw_table(&self) -> Vec<(i32, front::ty::Ty)> {
        vec![]
    }
}
