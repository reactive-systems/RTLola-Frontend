use super::*;
extern crate regex;

use crate::pacing_types::{
    parse_abstract_type, AbstractPacingType, Freq, PacingError, UnificationError,
};

use crate::rtltc::NodeId;
use biodivine_lib_bdd::{BddVariableSet, BddVariableSetBuilder};
use front::common_ir::Offset;
use front::hir::expression::{Expression, ExpressionKind};
use front::hir::modes::ir_expr::WithIrExpr;
use front::hir::modes::HirMode;
use front::hir::{Input, Output, Trigger};
use front::parse::Span;
use front::RTLolaHIR;
use rusttyc::types::AbstractTypeTable;
use rusttyc::{TcErr, TcKey, TypeChecker};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable(String);

impl rusttyc::TcVar for Variable {}

pub struct Context<'a, M>
where
    M: HirMode + WithIrExpr + 'static,
{
    pub(crate) hir: &'a RTLolaHIR<M>,
    pub(crate) tyc: TypeChecker<AbstractPacingType, Variable>,
    pub(crate) node_key: HashMap<NodeId, TcKey>,
    pub(crate) bdd_vars: BddVariableSet,
    pub(crate) key_span: HashMap<TcKey, Span>,
}

impl<'a, M> Context<'a, M>
where
    M: HirMode + WithIrExpr + 'static,
{
    pub(crate) fn new(hir: &'a RTLolaHIR<M>) -> Self {
        let mut bdd_var_builder = BddVariableSetBuilder::new();
        let mut node_key = HashMap::new();
        let mut tyc = TypeChecker::new();
        let key_span = HashMap::new();

        for input in hir.inputs() {
            bdd_var_builder.make_variable(&input.sr.in_ix().to_string()); //TODO
            let key = tyc.get_var_key(&Variable(input.name.clone()));
            node_key.insert(NodeId::SRef(input.sr), key);
            //key_span.insert(key, input.span);
        }
        for output in hir.outputs() {
            bdd_var_builder.make_variable(&(output.sr.out_ix() + hir.num_inputs()).to_string()); //TODO
            let key = tyc.get_var_key(&Variable(output.name.clone()));
            node_key.insert(NodeId::SRef(output.sr), key);
            //key_span.insert(key, output.span);
        }
        /*
        for ast_const in &ast.constants {
            bdd_var_builder.make_variable(&ast_const.id.to_string());
            let key = tyc.get_var_key(&Variable(ast_const.name.name.clone()));
            node_key.insert(ast_const.id, key);
            key_span.insert(key, ast_const.span);
        }
        */
        let bdd_vars = bdd_var_builder.build();

        Context {
            hir,
            tyc,
            node_key,
            bdd_vars,
            key_span,
        }
    }

    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<(), TcErr<AbstractPacingType>> {
        let ac =
            AbstractPacingType::Event(self.bdd_vars.mk_var_by_name(&input.sr.in_ix().to_string()));
        let input_key = self.node_key[&NodeId::SRef(input.sr)];
        self.tyc.impose(input_key.concretizes_explicit(ac))
    }

    /*
    pub(crate) fn constant_infer(
        &mut self,
        constant: &Constant,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Any;
        let const_key = self.node_key[&constant.id];
        self.tyc.impose(const_key.concretizes_explicit(ac))
    }
    */

    pub(crate) fn trigger_infer(
        &mut self,
        trigger: &Trigger,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        let ex_key = self.expression_infer(self.hir.expr(trigger.sr))?;
        let trigger_key = self.tyc.new_term_key();
        self.node_key.insert(NodeId::SRef(trigger.sr), trigger_key);
        //self.key_span.insert(trigger_key, trigger.span);
        self.tyc.impose(trigger_key.equate_with(ex_key))
    }

    pub(crate) fn output_infer(
        &mut self,
        output: &Output,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        // Type Expression
        let exp_key = self.expression_infer(&self.hir.expr(output.sr))?;
        let output_key = self.node_key[&NodeId::SRef(output.sr)];
        self.tyc
            .impose(output_key.concretizes_explicit(AbstractPacingType::Never))?;

        // Check if there is a type is annotated
        if let Some(expr) = self.hir.act_cond(output.sr) {
            let annotated_ac_key = self.tyc.new_term_key();
            self.node_key
                .insert(NodeId::Expr(expr.eid), annotated_ac_key);
            //self.key_span.insert(annotated_ac_key, output.extend.span);

            let annotated_ac =
                match parse_abstract_type(expr, &self.bdd_vars, self.hir.num_inputs()) {
                    Ok(b) => b,
                    Err(reason) => {
                        return Err(TcErr::Bound(
                            annotated_ac_key,
                            None,
                            UnificationError::Other(reason),
                        ));
                    }
                };

            // Bind key to parsed type
            self.tyc
                .impose(annotated_ac_key.has_exactly_type(annotated_ac))?;

            // Annotated type should be more concrete than inferred type
            self.tyc.impose(annotated_ac_key.concretizes(exp_key))?;

            // Output type is equal to declared type
            self.tyc.impose(output_key.concretizes(annotated_ac_key))
        } else {
            // Output type is equal to inferred type
            self.tyc.impose(output_key.concretizes(exp_key))
        }
    }

    pub(crate) fn expression_infer(
        &mut self,
        exp: &Expression,
    ) -> Result<TcKey, TcErr<AbstractPacingType>> {
        let term_key: TcKey = self.tyc.new_term_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::LoadConstant(_) | ExpressionKind::ParameterAccess(_, _) => {
                //constants have arbitrary pacing type
                self.tyc.impose(term_key.has_exactly_type(Any))?;
            }
            ExpressionKind::StreamAccess(sref, kind, args) => {
                use front::hir::expression::StreamAccessKind;
                let stream_key = self.node_key[&NodeId::SRef(*sref)];
                match kind {
                    StreamAccessKind::Sync => self.tyc.impose(term_key.equate_with(stream_key))?,
                    StreamAccessKind::Offset(off) => {
                        match off {
                            Offset::PastRealTimeOffset(_) | Offset::FutureRealTimeOffset(_) => {
                                // Real time offset are only allowed on timed streams.
                                self.tyc.impose(term_key.concretizes_explicit(
                                    AbstractPacingType::Periodic(Freq::Any),
                                ))?;
                                self.tyc.impose(term_key.concretizes(stream_key))?;
                            }
                            Offset::PastDiscreteOffset(_) | Offset::FutureDiscreteOffset(_) => {
                                self.tyc.impose(term_key.concretizes(stream_key))?;
                            }
                        }
                    }
                    StreamAccessKind::Hold => {
                        self.tyc.impose(term_key.concretizes_explicit(Any))?
                    }
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => {
                        self.tyc.impose(
                            term_key.concretizes_explicit(AbstractPacingType::Periodic(Freq::Any)),
                        )?;
                        // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
                    }
                };

                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.tyc.impose(term_key.concretizes(arg_key))?;
                }
            }
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr)?;
                let def_key = self.expression_infer(&*default)?;

                self.tyc.impose(term_key.is_meet_of(ex_key, def_key))?;
            }
            ExpressionKind::ArithLog(_, args) => match args.len() {
                2 => {
                    let left_key = self.expression_infer(&args[0])?;
                    let right_key = self.expression_infer(&args[1])?;

                    self.tyc.impose(term_key.is_meet_of(left_key, right_key))?;
                }
                1 => {
                    let ex_key = self.expression_infer(&args[0])?;

                    self.tyc.impose(term_key.equate_with(ex_key))?;
                }
                _ => unreachable!(),
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                let cond_key = self.expression_infer(&*condition)?;
                let cons_key = self.expression_infer(&*consequence)?;
                let alt_key = self.expression_infer(&*alternative)?;

                self.tyc
                    .impose(term_key.is_meet_of_all(&[cond_key, cons_key, alt_key]))?;
            }
            ExpressionKind::Tuple(elements) => {
                let ele_keys: Vec<TcKey> = elements
                    .iter()
                    .map(|e| self.expression_infer(&*e))
                    .collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>(
                )?;
                self.tyc.impose(term_key.is_meet_of_all(&ele_keys))?;
            }
            /*
            ExpressionKind::Field(exp, _ident) => {
                //TODO unused var
                let exp_key = self.expression_infer(&*exp)?;
                self.tyc.impose(term_key.equate_with(exp_key))?;
            }
            */
            /*
            ExpressionKind::Method(body, _, _, args) => {
                let body_key = self.expression_infer(&*body)?;
                let mut arg_keys: Vec<TcKey> =
                    args.iter()
                        .map(|e| self.expression_infer(&*e))
                        .collect::<Result<Vec<TcKey>, TcErr<AbstractPacingType>>>()?;
                arg_keys.push(body_key);
                self.tyc.impose(term_key.is_meet_of_all(&arg_keys))?;
            }
            */
            ExpressionKind::Function { args, .. } => {
                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.tyc.impose(term_key.concretizes(arg_key))?;
                }
            }
            ExpressionKind::TupleAccess(t, _) => {
                let exp_key = self.expression_infer(&*t)?;
                self.tyc.impose(term_key.equate_with(exp_key))?;
            }
            ExpressionKind::Widen(inner, _) => {
                let exp_key = self.expression_infer(&*inner)?;
                self.tyc.impose(term_key.equate_with(exp_key))?;
            }
        };
        self.node_key.insert(NodeId::Expr(exp.eid), term_key);
        //self.key_span.insert(term_key, exp.span);
        Ok(term_key)
    }

    pub(crate) fn post_process(
        hir: &RTLolaHIR<M>,
        nid_key: HashMap<NodeId, TcKey>,
        tt: &AbstractTypeTable<AbstractPacingType>,
    ) -> Vec<PacingError> {
        let mut res = vec![];
        // That every periodic stream has a frequency
        for output in hir.outputs() {
            let at = &tt[nid_key[&NodeId::SRef(output.sr)]];
            match at {
                AbstractPacingType::Periodic(Freq::Any) => {
                    res.push(PacingError::FreqAnnotationNeeded(Span::unknown()));
                    //TODO
                }
                AbstractPacingType::Never => {
                    //TOD
                    res.push(PacingError::NeverEval(Span::unknown()));
                    //TOD
                }
                _ => {}
            }
        }
        res
    }
}

#[cfg(test)]
/*
Todo:
- aggregations
 */
mod pacing_type_tests {
    use crate::pacing_types::{ActivationCondition, ConcretePacingType};
    use crate::LolaTypeChecker;
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
        let mut ltc = LolaTypeChecker::new(&spec, dec.clone(), &handler);
        ltc.pacing_type_infer();
        return handler.emitted_errors();
    }

    #[test]
    fn test_input_simple() {
        let spec = "input i: Int8";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(tt[&ast.constants[0].id], ConcretePacingType::Constant);
    }

    #[test]
    fn test_trigger_simple() {
        let spec = "input a: Int8\n trigger a == 42";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(ast.inputs[0].id);
        assert_eq!(tt[&ast.trigger[0].id], ConcretePacingType::Event(ac_a));
    }

    #[test]
    fn test_disjunction_annotated() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
    fn test_no_direct_access_possible() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := a";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_1hz_meet() {
        let spec =
            "input i: Int64\noutput a @ 5Hz := 42\noutput b @ 2Hz := 1337\noutput c := a + b";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&ast.outputs[2].id],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(1).unwrap()
            ))
        );
    }

    #[test]
    fn test_0_1hz_meet() {
        let spec =
            "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 0.3Hz := 1337\noutput c := a + b";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&ast.outputs[2].id],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
                Rational::from_f32(0.1).unwrap()
            ))
        );
    }

    #[test]
    fn test_annotated_freq() {
        let spec =
            "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 3Hz := 1337\noutput c @2Hz := a + b";
        assert_eq!(num_errors(spec), 1);
    }

    #[test]
    fn test_parametric_output() {
        let spec =
            "input i: UInt8\noutput x(a: UInt8, b: Bool): Int8 := i\noutput y := x(1, false)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let i_type = ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id));
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&ast.outputs[0].id], i_type);
        assert_eq!(tt[&ast.outputs[1].id], i_type);
    }

    #[test]
    fn test_parametric_output_parameter() {
        let spec = "input i: UInt8\noutput x(a: UInt8, b: Bool) @i := a";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\ninput i: UInt8\noutput o: Float32 @i := sin(2.0)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_tuple() {
        let spec = "input i: UInt8\noutput out: (Int8, Bool) @i:= (14, false)";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&ast.outputs[0].id],
            ConcretePacingType::Event(ActivationCondition::Stream(ast.inputs[0].id))
        );
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
        let (ast, dec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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
        let mut ltc = LolaTypeChecker::new(&ast, dec.clone(), &handler);
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

    #[test]
    fn test_timed() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_timed_faster() {
        let spec = "output o1: Bool @20Hz := false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_timed_incompatible() {
        let spec = "output o1: Bool @3Hz := false\noutput o2: Bool@10Hz:= o1";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_timed_binary() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1 && true";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x.hold().defaults(to: 0)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_offset_not_possible() {
        let spec = "input x: UInt8\n output a := x\noutput b := a.offset(by: 1s)";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_realtime_offset_possible() {
        let spec = "input x: UInt8\n output a @1Hz := x.hold()\noutput b := a.offset(by: 1s)";
        assert_eq!(0, num_errors(spec));
    }
}
