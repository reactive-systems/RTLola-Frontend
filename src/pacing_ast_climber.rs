use super::*;
extern crate regex;

use crate::pacing_types::{
    parse_abstract_type, AbstractPacingType, ActivationCondition, ConcretePacingType, Freq,
};
use bimap::{BiMap, Overwritten};
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
    pub(crate) node_key: BiMap<NodeId, TcKey>,
    pub(crate) bdd_vars: BddVariableSet,
}

impl<'a> Context<'a> {
    pub fn new(ast: &RTLolaAst, decl: &'a DeclarationTable) -> Context<'a> {
        let mut bdd_var_builder = BddVariableSetBuilder::new();
        let mut node_key = BiMap::new();
        let mut tyc = TypeChecker::new();

        for input in &ast.inputs {
            bdd_var_builder.make_variable(&input.id.to_string());
            let key = tyc.get_var_key(&Variable(input.name.name.clone()));
            node_key.insert(input.id, key);
        }
        for output in &ast.outputs {
            bdd_var_builder.make_variable(&output.id.to_string());
            let key = tyc.get_var_key(&Variable(output.name.name.clone()));
            node_key.insert(output.id, key);
        }
        for ast_const in &ast.constants {
            bdd_var_builder.make_variable(&ast_const.id.to_string());
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
        let ac = AbstractPacingType::Event(self.bdd_vars.mk_var_by_name(&input.id.to_string()));
        let input_key = self.node_key.get_by_left(&input.id).unwrap();
        self.tyc.impose(input_key.concretizes_explicit(ac))
    }

    pub fn constant_infer(&mut self, constant: &Constant) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Event(self.bdd_vars.mk_true());
        let const_key = self.node_key.get_by_left(&constant.id).unwrap();
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
        let declared_ac =
            match parse_abstract_type(output.extend.expr.as_ref(), &self.bdd_vars, self.decl) {
                Ok(b) => b,
                Err((reason, span)) => {
                    // Todo: Handle properly once the interface is done
                    unimplemented!("{}", reason);
                }
            };
        let declared_ac_key = self.tyc.new_term_key();
        self.tyc
            .impose(declared_ac_key.concretizes_explicit(declared_ac))?;
        self.node_key.insert(output.extend.id, declared_ac_key);

        // Type Expression
        let exp_key = self.expression_infer(&output.expression)?;

        // Infer resulting type
        let output_key = self.node_key.get_by_left(&output.id).unwrap();
        self.tyc
            .impose(output_key.is_meet_of(declared_ac_key, exp_key))?;

        // Todo: How to check implication?
        self.tyc.impose(output_key.equate_with(declared_ac_key))
    }

    pub fn expression_infer(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey, TcErr<AbstractPacingType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::Lit(_) => {
                let literal_type = Any;
                self.tyc
                    .impose(term_key.concretizes_explicit(literal_type))?;
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
                let key = self.node_key.get_by_left(&node_id).unwrap();
                self.tyc.impose(term_key.equate_with(*key))?;
            }
            ExpressionKind::StreamAccess(ex, kind) => {
                use front::ast::StreamAccessKind::*;
                let ex_key = self.expression_infer(ex)?;
                match kind {
                    Optional | Sync => self.tyc.impose(term_key.equate_with(ex_key))?,
                    Hold => self.tyc.impose(term_key.concretizes_explicit(Any))?,
                };
            }
            ExpressionKind::Default(ex, default) => {
                let ex_key = self.expression_infer(&*ex)?;
                let def_key = self.expression_infer(&*default)?;

                self.tyc.impose(term_key.is_meet_of(ex_key, def_key))?;
            }
            ExpressionKind::Offset(expr, _) => {
                let ex_key = self.expression_infer(&*expr)?;

                self.tyc.impose(term_key.equate_with(ex_key))?;
            }
            ExpressionKind::SlidingWindowAggregation { .. } => {
                self.tyc.impose(
                    term_key.concretizes_explicit(AbstractPacingType::Periodic(Freq::Any)),
                )?;
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

                self.tyc
                    .impose(term_key.is_meet_of_all(&vec![cond_key, cons_key, alt_key]))?;
            }
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(elements) => {
                let ele_keys: Vec<TcKey> = elements
                    .iter()
                    .map(|e| self.expression_infer(&*e))
                    .collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>(
                )?;
                self.tyc.impose(term_key.is_meet_of_all(&ele_keys))?;
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(body, _, _, args) => {
                let body_key = self.expression_infer(&*body)?;
                let mut arg_keys: Vec<TcKey> =
                    args.iter()
                        .map(|e| self.expression_infer(&*e))
                        .collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>()?;
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
                        let output_key = self.node_key.get_by_left(&out.id).unwrap();
                        self.tyc.impose(term_key.concretizes(*output_key))?;
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

#[cfg(test)]
/*
Todo:
- Constant not constant
- aggregations
- parameters
 */
mod pacing_type_tests {
    use crate::pacing_types::{AbstractPacingType, ActivationCondition, ConcretePacingType};
    use crate::LolaTypChecker;
    use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};
    use front::analysis::naming::Declaration;
    use front::parse::NodeId;
    use front::parse::SourceMapper;
    use front::reporting::Handler;
    use front::RTLolaAst;
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    fn setup_ast(spec: &str) -> (RTLolaAst, HashMap<NodeId, Declaration>, Handler) {
        let handler = front::reporting::Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let spec: RTLolaAst =
            match front::parse::parse(spec, &handler, front::FrontendConfig::default()) {
                Ok(s) => s,
                Err(e) => panic!("Spec {} cannot be parsed: {}", spec, e),
            };
        let mut na = front::analysis::naming::NamingAnalysis::new(
            &handler,
            front::FrontendConfig::default(),
        );
        let dec = na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        (spec, dec, handler)
    }

    fn num_errors(spec: &str) -> usize {
        let (spec, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&spec, dec.clone(), &handler);
        ltc.pacing_type_infer();
        return handler.emitted_errors();
    }

    #[test]
    fn test_input_simple() {
        let spec = "input i: Int8";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&ast.inputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_output_simple() {
        let spec = "input a: Int8\n input b: Int8 \n output o := a + b";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(ast.inputs[0].id);
        let ac_b = ActivationCondition::Stream(ast.inputs[1].id);
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_constant_simple() {
        let spec = "constant c: UInt8 := -2";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&ast.constants[0].id],
            ConcretePacingType::Event(ActivationCondition::True)
        );
    }

    #[test]
    fn test_trigger_simple() {
        let spec = "input a: Int8\n trigger a == 42";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(ast.inputs[0].id);
        assert_eq!(tt[&ast.trigger[0].id], ConcretePacingType::Event(ac_a));
    }

    #[test]
    fn test_disjunction_annotated() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let ac_a = ActivationCondition::Stream(ast.inputs[0].id);
        let ac_b = ActivationCondition::Stream(ast.inputs[1].id);
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Disjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_frequency_simple() {
        let spec = "output a: UInt8 @10Hz := 0";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Periodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(10).unwrap()
            ))
        );
    }

    #[test]
    fn test_frequency_conjunction() {
        let spec = "output a: Int32 @10Hz := 0\noutput b: Int32 @5Hz := 0\noutput x := a+b";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&ast.outputs[2].id],
            ConcretePacingType::Periodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(5).unwrap()
            ))
        );
    }

    #[test]
    fn test_get_not_possible() {
        // it should not be possible to use get with RealTime and EventBased streams
        let spec = "input a: Int32\noutput x: Int32 @ 1Hz := 0\noutput y:Int32 @ a := x.get().defaults(to: 0)";
        assert_eq!(num_errors(spec), 1);
        let spec = "input a: Int32\noutput x: Int32 @ a := 0\noutput y:Int32 @ 1Hz := x.get().defaults(to: 0)";
        assert_eq!(num_errors(spec), 1);
    }

    #[test]
    fn test_normalization_event_streams() {
        let spec = "input a: Int32\ninput b: Int32\ninput c: Int32\noutput x := a + b\noutput y := x + x + c";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        //  input `a` has NodeId =  1
        let a_id = NodeId::new(1);
        //  input `b` has NodeId =  3
        let b_id = NodeId::new(3);
        //  input `c` has NodeId =  5
        let c_id = NodeId::new(5);
        // output `x` has NodeId = 11
        let x_id = NodeId::new(11);
        // output `y` has NodeId = 19
        let y_id = NodeId::new(19);

        assert_eq!(
            tt[&x_id],
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(a_id),
                ActivationCondition::Stream(b_id)
            ]))
        );

        assert_eq!(
            tt[&y_id],
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![
                ActivationCondition::Stream(a_id),
                ActivationCondition::Stream(b_id),
                ActivationCondition::Stream(c_id)
            ]))
        );
    }

    #[test]
    fn test_realtime_stream_integer_offset() {
        let spec = "output b @2Hz := b[-1].defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_faster() {
        let spec =
            "output a @4Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // equivalent to b[-500ms].defaults(to: 0) + a[-250ms].defaults(to: 0)
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_incompatible() {
        let spec =
            "output a @3Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_sample_and_hold() {
        let spec = "
            output a @3Hz := 0
            output a_offset := a[-1].defaults(to: 0)
            output b @2Hz := b[-1].defaults(to: 0) + a_offset.hold().defaults(to: 0)
        ";
        // workaround using sample and hold
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    #[ignore] // Fix me
    fn test_no_direct_access_possible() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := a";
        assert_eq!(1, num_errors(spec));
    }
}
