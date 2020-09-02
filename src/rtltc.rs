use super::*;

use crate::pacing_ast_climber::Context as PacingContext;
use crate::pacing_types::{emit_error, ConcretePacingType};
use crate::value_ast_climber::ValueContext;
use crate::value_types::IConcreteType;
use front::analysis::naming::DeclarationTable;
use front::ast::RTLolaAst;
use front::parse::NodeId;
use front::reporting::{Handler, LabeledSpan};
use rusttyc::types::ReifiedTypeTable;
use rusttyc::TcErr;
use std::collections::HashMap;

pub struct LolaTypeChecker<'a> {
    pub(crate) ast: RTLolaAst,
    pub(crate) declarations: DeclarationTable,
    pub(crate) handler: &'a Handler,
}

impl<'a> LolaTypeChecker<'a> {
    pub fn new(spec: &RTLolaAst, declarations: DeclarationTable, handler: &'a Handler) -> Self {
        LolaTypeChecker {
            ast: spec.clone(),
            declarations,
            handler,
        }
    }

    pub fn check(&mut self) {
        if let Some(_table) = self.pacing_type_infer() {
            //TODO
        };
        //TODO imports
        match self.value_type_infer() {
            Ok(_map) => {}
            Err(_e) => {} //TODO,
        };
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
        let nid_key = ctx.node_key.clone();
        let tt = match ctx.tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                emit_error(&e, self.handler, &ctx.bdd_vars, &ctx.key_span, &input_names);
                return None;
            }
        };

        if self.handler.contains_error() {
            return None;
        }

        for pe in PacingContext::post_process(nid_key, &self.ast, &tt) {
            pe.emit(self.handler);
        }
        if self.handler.contains_error() {
            return None;
        }

        let key_span = ctx.key_span.clone();
        let ctt: HashMap<NodeId, ConcretePacingType> = ctx
            .node_key
            .iter()
            .filter_map(|(id, key)| {
                match ConcretePacingType::from_abstract(tt[*key].clone(), &vars) {
                    Ok(ct) => Some((*id, ct)),
                    Err(e) => {
                        e.emit_with_span(self.handler, key_span[key]);
                        None
                    }
                }
            })
            .collect();

        if self.handler.contains_error() {
            return None;
        }
        Some(ctt)
    }

    pub(crate) fn value_type_infer(&self) -> Result<HashMap<NodeId, IConcreteType>, String> {
        //let value_tyc = rusttyc::TypeChecker::new();

        let mut ctx = ValueContext::new(&self.ast, self.declarations.clone(), self.handler);

        for input in &self.ast.inputs {
            if let Err(e) = ctx.input_infer(input) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        for constant in &self.ast.constants {
            if let Err(e) = ctx.constant_infer(constant) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        for output in &self.ast.outputs {
            if let Err(e) = ctx.output_infer(output) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        for trigger in &self.ast.trigger {
            if let Err(e) = ctx.trigger_infer(trigger) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        let tt_r = ctx.tyc.clone().type_check();
        if let Err(tc_err) = tt_r {
            let (error_key, error_key_2) = match tc_err.clone() {
                //TODO improve
                TcErr::ChildAccessOutOfBound(key, _ty, _n) => (key, None),
                TcErr::KeyEquation(k1, k2, _msg) => (k1, Some(k2)),
                TcErr::Bound(key, key2, _msg) => (key, key2),
                TcErr::ExactTypeViolation(key, _bound) => (key, None),
                TcErr::ConflictingExactBounds(key, _bound1, _bound2) => (key, None),
            };
            self.handler.error_with_span(
                "Result inference error",
                LabeledSpan::new(ctx.key_span[&error_key], "Todo", true),
            );
            if let Some(k) = error_key_2 {
                self.handler.warn_with_span(
                    "Result inference error - connected warning",
                    LabeledSpan::new(ctx.key_span[&k], "Todo", false),
                );
            }
            let msg: String = ctx.handle_error(tc_err);
            return Err(msg);
        }
        let tt = tt_r.expect("Ensured by previous cases");
        let bm = ctx.node_key;
        for (nid, k) in bm.iter() {
            //DEBUGG
            println!("{:?}", (*nid, tt[*k].clone()));
        }
        let rtt_r = tt.try_reified();
        if rtt_r.is_err() {
            return Err("TypeTable not reifiable: ValueType not constrained enough".to_string());
        }
        let rtt: ReifiedTypeTable<IConcreteType> = rtt_r.unwrap();
        let mut result_map = HashMap::new();
        for (nid, k) in bm.iter() {
            result_map.insert(*nid, rtt[*k].clone());
        }
        Ok(result_map)
    }
}
