use super::*;
extern crate regex;

use crate::pacing_types::{
    parse_abstract_type, AbstractPacingType, ActivationCondition, ConcretePacingType, Freq,
    UnificationError,
};
use bimap::{BiMap, Overwritten};
use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{Constant, Expression, Input, Output, Trigger};
use front::ast::{ExpressionKind, RTLolaAst};
use front::parse::{NodeId, Span};
use front::reporting::{Handler, LabeledSpan};
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
    pub(crate) key_span: HashMap<TcKey, Span>,
}

impl<'a> Context<'a> {
    pub(crate) fn new(ast: &RTLolaAst, decl: &'a DeclarationTable) -> Context<'a> {
        let mut bdd_var_builder = BddVariableSetBuilder::new();
        let mut node_key = HashMap::new();
        let mut tyc = TypeChecker::new();
        let mut key_span = HashMap::new();

        for input in &ast.inputs {
            bdd_var_builder.make_variable(&input.id.to_string());
            let key = tyc.get_var_key(&Variable(input.name.name.clone()));
            node_key.insert(input.id, key);
            key_span.insert(key, input.span);
        }
        for output in &ast.outputs {
            bdd_var_builder.make_variable(&output.id.to_string());
            let key = tyc.get_var_key(&Variable(output.name.name.clone()));
            node_key.insert(output.id, key);
            key_span.insert(key, output.span);
        }
        for ast_const in &ast.constants {
            bdd_var_builder.make_variable(&ast_const.id.to_string());
            let key = tyc.get_var_key(&Variable(ast_const.name.name.clone()));
            node_key.insert(ast_const.id, key);
            key_span.insert(key, ast_const.span);
        }
        let bdd_vars = bdd_var_builder.build();

        Context {
            tyc,
            decl,
            node_key,
            bdd_vars,
            key_span,
        }
    }

    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Event(self.bdd_vars.mk_var_by_name(&input.id.to_string()));
        let input_key = self.node_key[&input.id];
        self.tyc.impose(input_key.concretizes_explicit(ac))
    }

    pub(crate) fn constant_infer(
        &mut self,
        constant: &Constant,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Any;
        let const_key = self.node_key[&constant.id];
        self.tyc.impose(const_key.concretizes_explicit(ac))
    }

    pub(crate) fn trigger_infer(
        &mut self,
        trigger: &Trigger,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        let ex_key = self.expression_infer(&trigger.expression)?;
        let trigger_key = self.tyc.new_term_key();
        self.node_key.insert(trigger.id, trigger_key);
        self.key_span.insert(trigger_key, trigger.span);
        self.tyc.impose(trigger_key.equate_with(ex_key))
    }

    pub(crate) fn output_infer(
        &mut self,
        output: &Output,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        // Type Expression
        let exp_key = self.expression_infer(&output.expression)?;
        let output_key = self.node_key[&output.id];

        // Check if there is a type is annotated
        if let Some(expr) = output.extend.expr.as_ref() {
            let annotated_ac_key = self.tyc.new_term_key();
            self.node_key.insert(output.extend.id, annotated_ac_key);
            self.key_span.insert(annotated_ac_key, output.extend.span);

            let annotated_ac = match parse_abstract_type(expr, &self.bdd_vars, self.decl) {
                Ok(b) => b,
                Err((reason, span)) => {
                    return Err(TcErr::TypeBound(
                        annotated_ac_key,
                        UnificationError::Other(reason),
                    ));
                }
            };

            // Bind key to parsed type
            // Todo: should be explicit bound
            self.tyc
                .impose(annotated_ac_key.concretizes_explicit(annotated_ac))?;

            // Annotated type should be more concrete than inferred type
            self.tyc.impose(annotated_ac_key.concretizes(exp_key))?;

            // Output type is equal to declared type
            self.tyc.impose(output_key.equate_with(annotated_ac_key))
        } else {
            // Output type is equal to inferred type
            self.tyc.impose(output_key.equate_with(exp_key))
        }
    }

    pub(crate) fn expression_infer(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey, TcErr<AbstractPacingType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::Lit(_) => {
                // Todo: Should be explicit bound
                self.tyc.impose(term_key.concretizes_explicit(Any))?;
            }
            ExpressionKind::Ident(_) => {
                let decl = &self.decl[&exp.id];
                if let Declaration::Param(_) = decl {
                    self.tyc.impose(term_key.concretizes_explicit(Any))?;
                } else {
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
                let left_key = self.expression_infer(&*left)?;
                let right_key = self.expression_infer(&*right)?;

                self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
            }
            ExpressionKind::Unary(_, expr) => {
                let ex_key = self.expression_infer(&*expr)?;

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
            ExpressionKind::Field(exp, iden) => {
                let exp_key = self.expression_infer(&*exp)?;
                self.tyc.impose(term_key.equate_with(exp_key))?;
            }
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
        self.key_span.insert(term_key, exp.span);
        Ok(term_key)
    }

    pub(crate) fn post_process(
        ast: &RTLolaAst,
        ctt: &HashMap<NodeId, ConcretePacingType>,
    ) -> Result<(), (String, Span)> {
        // That every periodic stream has a frequency
        for output in &ast.outputs {
            let ct = &ctt[&output.id];
            match ct {
                ConcretePacingType::Periodic => {
                    return Err((
                        "This stream is missing a frequency annotation.".to_string(),
                        output.span,
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
/*
Todo:
- aggregations
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
        assert_eq!(tt[&ast.constants[0].id], ConcretePacingType::Constant);
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
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
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
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
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

    #[test]
    fn test_parametric_output() {
        let spec =
            "input i: UInt8\noutput x(a: UInt8, b: Bool): Int8 := i\noutput y := x(1, false)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let i_type = ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id));
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&ast.outputs[0].id], i_type);
        assert_eq!(tt[&ast.outputs[1].id], i_type);
    }

    #[test]
    fn test_parametric_output_parameter() {
        let spec = "output x(a: UInt8, b: Bool) := a";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&ast.outputs[0].id], ConcretePacingType::Constant);
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\noutput o: Float32 := sin(2.0)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&ast.outputs[0].id], ConcretePacingType::Constant);
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&ast.outputs[0].id], ConcretePacingType::Constant);
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(5).unwrap()
            ))
        );
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_invalid_op_in_ac() {
        let spec = "input in: Int8\n output out: Int16 @!in := 5";
        assert_eq!(1, num_errors(spec));
    }
}
