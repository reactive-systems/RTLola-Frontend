use super::*;

use crate::pacing_ast_climber::Context as PacingContext;
use crate::value_ast_climber::ValueContext;
use crate::value_types::IAbstractType;
use front::analysis::naming::DeclarationTable;
use front::ast::LolaSpec;

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

    fn pacing_type_infer(&mut self) {
        let mut ctx = PacingContext::new(&self.ast, &self.declarations);

        for input in &self.ast.inputs {
            ctx.input_infer(input);
        }

        for constant in &self.ast.constants {
            ctx.constant_infer(constant);
        }

        for output in &self.ast.outputs {
            ctx.output_infer(output);
        }

        for trigger in &self.ast.trigger {
            ctx.trigger_infer(trigger);
        }

        let tt = ctx.tyc.type_check();
    }

    fn value_type_infer(&self) {
        //let value_tyc = rusttyc::TypeChecker::new();

        let mut ctx = ValueContext::new(&self.ast, self.declarations.clone());

        for constant in &self.ast.constants {
            ctx.constant_infer(&constant);
        }

        for output in &self.ast.outputs {
            ctx.expression_infer(&output.expression, None);
        }

        for trigger in &self.ast.trigger {
            ctx.expression_infer(&trigger.expression, Some(IAbstractType::Bool));
        }

        let tt = ctx.tyc.type_check();
    }

    pub fn generate_raw_table(&self) -> Vec<(i32, front::ty::Ty)> {
        vec![]
    }
}
