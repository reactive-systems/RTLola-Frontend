use super::*;

use crate::pacing_ast_climber::Context as PacingContext;
use crate::value_ast_climber::ValueContext;
use crate::value_types::IAbstractType;
use front::analysis::naming::DeclarationTable;
use front::ast::RTLolaAst;
use front::reporting::Handler;

#[derive()]
pub struct LolaTypChecker<'a> {
    pub(crate) ast: RTLolaAst,
    pub(crate) declarations: DeclarationTable,
    pub(crate) handler: &'a Handler,
}

impl<'a> LolaTypChecker<'a> {
    pub fn new(spec: &RTLolaAst, declarations: DeclarationTable, handler: &'a Handler) -> Self {
        LolaTypChecker {
            ast: spec.clone(),
            declarations: declarations.clone(),
            handler
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


#[cfg(test)]
mod value_type_tests {
    use std::path::PathBuf;
    use crate::LolaTypChecker;
    use front::parse::SourceMapper;

    fn check_set_up(spec: &str) -> usize {

        let handler = front::reporting::Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let spec = match front::parse::parse(spec,&handler, front::FrontendConfig::default()) {
            Ok(s) => s,
            Err(e) => panic!("Spech {} cannot be parsed: {}",spec,e),
        };

        let mut na = front::analysis::naming::NamingAnalysis::new(&handler, front::FrontendConfig::default());
        let mut dec = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut ltc = LolaTypChecker::new(&spec, dec, &handler);
        ltc.check();
        handler.emitted_errors()

    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, check_set_up(spec));
    }
}
