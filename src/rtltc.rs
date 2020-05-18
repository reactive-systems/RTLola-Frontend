use super::*;

use crate::astclimb::Variable;
use crate::types::IAbstractType;
use crate::astclimb::Context;
use front::analysis::naming::DeclarationTable;
use front::ast::LolaSpec;
use rusttyc::{Abstract, TypeChecker};
use std::collections::HashSet;

#[derive()]
pub struct LolaTypChecker<'a> {
    pub(crate) ast: LolaSpec,
    pub(crate) declarations: DeclarationTable<'a>,
}

impl<'a> LolaTypChecker<'a> {
    pub fn new(spec: &LolaSpec, declarations: DeclarationTable<'a>) -> Self {
        LolaTypChecker {
            ast: spec.clone(),
            declarations: declarations.clone(),
        }
    }

    pub fn check(&mut self) {
        //TODO imports
        self.value_type_infer();
        self.pacing_type_infer();

    }

    fn pacing_type_infer(){
        //TODO insert florians code

    }


    fn value_type_infer(&self) {
        let value_tyc = rusttyc::TypeChecker::new();

        let mut ctx = Context::new(tyc, self.decl);

        for constant in self.ast.constants {
            ctx.constant_infer(&constant);
        }

        for output in self.ast.outputs {
            ctx.expression_infer(&output.expression, None);
        }

        for trigger in self.ast.trigger {
            ctx.expression_infer(&trigger.expression, Some(IAbstractType::Bool));
        }

    }

    pub fn generate_raw_table(&self) -> Vec<(i32, front::ty::Ty)> {
        vec![]
    }
}
