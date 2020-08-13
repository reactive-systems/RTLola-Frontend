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

    fn value_type_infer(&self) -> Result<HashMap<NodeId, IConcreteType>, String> {
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

#[cfg(test)]
mod value_type_tests {
    use crate::value_types::IConcreteType;
    use crate::LolaTypChecker;
    use front::analysis::naming::Declaration;
    use front::parse::NodeId;
    use front::parse::SourceMapper;
    use front::reporting::Handler;
    use front::RTLolaAst;
    use rusttyc::types::{Abstract, Generalizable, ReifiedTypeTable};
    use std::collections::hash_map::RandomState;
    use std::collections::HashMap;
    use std::path::PathBuf;

    struct Test_Box {
        pub spec: RTLolaAst,
        pub dec: HashMap<NodeId, Declaration>,
        pub handler: Handler,
    }

    fn setup_ast(spec: &str) -> Test_Box {
        let handler = front::reporting::Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let spec: RTLolaAst =
            match front::parse::parse(spec, &handler, front::FrontendConfig::default()) {
                Ok(s) => s,
                Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
            };
        let mut na = front::analysis::naming::NamingAnalysis::new(
            &handler,
            front::FrontendConfig::default(),
        );
        let mut dec = na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        Test_Box { spec, dec, handler }
    }

    fn complete_check(spec: &str) -> usize {
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
        ltc.check();
        test_box.handler.emitted_errors()
    }

    fn check_value_type(spec: &str) -> (Test_Box, HashMap<NodeId, IConcreteType>) {
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
        let tt_result = ltc.value_type_infer();
        if let Err(ref e) = tt_result {
            eprintln!("{}", e.clone());
        }
        assert!(tt_result.is_ok(), "Expect Valid Input");
        let tt = tt_result.expect("");
        (test_box, tt)
    }

    fn check_expect_error(spec: &str) -> Test_Box {
        let test_box = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&test_box.spec, test_box.dec.clone(), &test_box.handler);
        let tt_result = ltc.value_type_infer();
        dbg!(&tt_result);
        assert!(tt_result.is_err());
        test_box
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn direct_implication() {
        let spec = "input i: Int8\noutput o := i";
        let (tb, result_map) = check_value_type(spec);
        let input_id = tb.spec.inputs[0].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(result_map[&input_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    fn integer_addition() {
        let spec = "input i: Int8\ninput i1: Int16\noutput o := i + i1";
        let (tb, result_map) = check_value_type(spec);
        let input_i_id = tb.spec.inputs[0].id;
        let input_i1_id = tb.spec.inputs[1].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(result_map[&input_i_id], IConcreteType::Integer16); //TODO FIXME due to explicit widening: want Int8 here
        assert_eq!(result_map[&input_i1_id], IConcreteType::Integer16);
        assert_eq!(result_map[&output_id], IConcreteType::Integer16);
        assert_eq!(0, complete_check(spec));
    }

    #[test]
    #[ignore] //naming analysis fails
    fn parametric_input() {
        let spec = "input i(a: Int8, b: Bool): Int8\noutput o := i(1,false)[0].defaults(to: 42)";
        let (tb, result_map) = check_value_type(spec);
        let input_id = tb.spec.inputs[0].id;
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&input_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_x() {
        let spec = "output x(a: UInt8, b: Bool): Int8 := 1";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
    }

    #[test]
    fn parametric_declaration_param_infer() {
        let spec = "output x(a: UInt8, b: Bool) := a";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        //assert_eq!(0, complete_check(spec)); //TODO FIXME: pacing types cant handle params in expr
        assert_eq!(0, tb.handler.emitted_errors());
        assert_eq!(result_map[&output_id], IConcreteType::UInteger8);
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 := 1 output y := x(1, false)";
        let (tb, result_map) = check_value_type(spec);
        let output_id = tb.spec.outputs[0].id;
        let output_2_id = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&output_id], IConcreteType::Integer8);
        assert_eq!(result_map[&output_2_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Float32);
    }

    #[test]
    fn simple_const_float16() {
        let spec = "constant c: Float16 := 2.1";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Float32);
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3";
        let (tb, result_map) = check_value_type(spec);
        let cons_id = tb.spec.constants[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&cons_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn simple_valid_coersion() {
        //TODO does not check output type, only validity
        for spec in &[
            "constant c: Int8 := 1\noutput o: Int32 := c",
            "constant c: UInt16 := 1\noutput o: UInt64 := c",
            "constant c: Float32 := 1.0\noutput o: Float64 := c",
        ] {
            let (tb, result_map) = check_value_type(spec);
            assert_eq!(0, complete_check(spec));
        }
    }

    #[test]
    fn simple_invalid_coersion() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 := c";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn simple_trigger() {
        let spec = "trigger false";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.spec.trigger[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&tr_id], IConcreteType::Bool);
    }

    #[test]
    fn simple_trigger_message() {
        let spec = "trigger false \"alert always\"";
        let (tb, result_map) = check_value_type(spec);
        let tr_id = tb.spec.trigger[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&tr_id], IConcreteType::Bool);
    }

    #[test]
    fn faulty_trigger() {
        let spec = "trigger 1";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 := 3 + 5";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_binary_input() {
        let spec = "input i: Int8\noutput o: Int8 := 3 + i";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let in_id = tb.spec.inputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
        assert_eq!(result_map[&in_id], IConcreteType::Integer8);
    }

    #[test]
    fn simple_unary() {
        let spec = "output o := !false \n\
                           output u: Bool := !false";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        let out_id_2 = tb.spec.outputs[1].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Bool);
        assert_eq!(result_map[&out_id_2], IConcreteType::Bool);
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool := !3";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 := false + 2.5";
        let tb = check_expect_error(spec);
        assert_eq!(1, complete_check(spec));
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 := if false then 1 else 2";
        let (tb, result_map) = check_value_type(spec);
        let out_id = tb.spec.outputs[0].id;
        assert_eq!(0, complete_check(spec));
        assert_eq!(result_map[&out_id], IConcreteType::Integer8);
    }
}
