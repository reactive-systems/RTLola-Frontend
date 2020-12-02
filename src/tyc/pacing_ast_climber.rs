use super::*;
extern crate regex;

use crate::tyc::pacing_types::{AbstractPacingType, ActivationCondition, Freq, PacingError};

use crate::tyc::rtltc::NodeId;
use crate::common_ir::Offset;
use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::modes::ir_expr::WithIrExpr;
use crate::hir::modes::HirMode;
use crate::hir::{Input, Output, Trigger, AC};
use crate::reporting::Span;
use crate::RTLolaHIR;
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
    pub(crate) key_span: HashMap<TcKey, Span>,
}

impl<'a, M> Context<'a, M>
where
    M: HirMode + WithIrExpr + 'static,
{
    pub(crate) fn new(hir: &'a RTLolaHIR<M>) -> Self {
        let mut node_key = HashMap::new();
        let mut tyc = TypeChecker::new();
        let mut key_span = HashMap::new();

        for input in hir.inputs() {
            let key = tyc.get_var_key(&Variable(input.name.clone()));
            node_key.insert(NodeId::SRef(input.sr), key);
            key_span.insert(key, input.span.clone());
        }
        for output in hir.outputs() {
            let key = tyc.get_var_key(&Variable(output.name.clone()));
            node_key.insert(NodeId::SRef(output.sr), key);
            key_span.insert(key, output.span.clone());
        }

        Context {
            hir,
            tyc,
            node_key,
            key_span,
        }
    }

    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<(), TcErr<AbstractPacingType>> {
        let ac = AbstractPacingType::Event(ActivationCondition::Stream(input.sr));
        let input_key = self.node_key[&NodeId::SRef(input.sr)];
        self.tyc.impose(input_key.concretizes_explicit(ac))
    }

    pub(crate) fn trigger_infer(
        &mut self,
        trigger: &Trigger,
    ) -> Result<(), TcErr<AbstractPacingType>> {
        let ex_key = self.expression_infer(self.hir.expr(trigger.sr))?;
        let trigger_key = self.tyc.new_term_key();
        self.node_key.insert(NodeId::SRef(trigger.sr), trigger_key);
        self.key_span.insert(trigger_key, trigger.span.clone());
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
        if let Some(ac) = &output.activation_condition {
            let annotated_ty_key = self.tyc.new_term_key();
            let annotated_ty = match ac {
                AC::Frequency { span, value } => {
                    self.key_span.insert(annotated_ty_key, span.clone());
                    AbstractPacingType::Periodic(Freq::Fixed(*value))
                }
                AC::Expr(eid) => {
                    let expr = self.hir.expression(*eid);
                    self.node_key
                        .insert(NodeId::Expr(expr.eid), annotated_ty_key);
                    self.key_span.insert(annotated_ty_key, expr.span.clone());
                    AbstractPacingType::Event(
                        ActivationCondition::parse(expr)
                            .map_err(|pe| TcErr::Bound(annotated_ty_key, None, pe))?,
                    )
                }
            };
            // Bind key to parsed type
            self.tyc
                .impose(annotated_ty_key.has_exactly_type(annotated_ty))?;

            // Annotated type should be more concrete than inferred type
            self.tyc.impose(annotated_ty_key.concretizes(exp_key))?;

            // Output type is equal to declared type
            self.tyc.impose(output_key.equate_with(annotated_ty_key))
        } else {
            // Output type is equal to inferred type
            self.tyc.impose(output_key.concretizes(exp_key))
        }
        /*
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
        */
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
                use crate::common_ir::StreamAccessKind;
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
        self.key_span.insert(term_key, exp.span.clone());
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
                    res.push(PacingError::FreqAnnotationNeeded(output.span.clone()));
                }
                AbstractPacingType::Never => {
                    res.push(PacingError::NeverEval(output.span.clone()));
                }
                _ => {}
            }
        }
        //Check that trigger expression does not access parameterized stream
        //for trigger in hir.triggers() {
        //    let at = &tt[nid_key[&NodeId::SRef(trigger.sr)]];
        //    if at.spawn != (AbstractPacingType::Any, AbstractExpressionType::Any)
        //        || at.filter != AbstractExpressionType::Any
        //        || at.close != AbstractExpressionType::Any
        //    {
        //        res.push(PacingError::ParameterizedExpr(trigger.span.clone()));
        //    }
        //}
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::pacing_types::{ActivationCondition, ConcretePacingType};
    use crate::rtltc::NodeId;
    use crate::LolaTypeChecker;
    use front::common_ir::StreamReference;
    use front::hir::modes::IrExpression;
    use front::hir::RTLolaHIR;
    use front::reporting::Handler;
    use front::RTLolaAst;
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use std::path::PathBuf;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    fn setup_ast(spec: &str) -> (RTLolaHIR<IrExpression>, Handler) {
        let handler = front::reporting::Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst =
            match front::parse::parse(spec, &handler, front::FrontendConfig::default()) {
                Ok(s) => s,
                Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
            };
        let hir = front::hir::RTLolaHIR::<IrExpression>::from_ast(
            ast,
            &handler,
            &front::FrontendConfig::default(),
        );
        (hir, handler)
    }

    fn num_errors(spec: &str) -> usize {
        let (spec, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&spec, &handler);
        ltc.pacing_type_infer();
        return handler.emitted_errors();
    }

    fn get_sr_for_name(hir: &RTLolaHIR<IrExpression>, name: &str) -> StreamReference {
        if let Some(i) = hir.get_input_with_name(name) {
            i.sr
        } else {
            hir.get_output_with_name(name).unwrap().sr
        }
    }
    fn get_node_for_name(hir: &RTLolaHIR<IrExpression>, name: &str) -> NodeId {
        NodeId::SRef(get_sr_for_name(hir, name))
    }

    #[test]
    fn test_input_simple() {
        let spec = "input i: Int8";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "i")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_output_simple() {
        let spec = "input a: Int8\n input b: Int8 \n output o := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        let ac_b = ActivationCondition::Stream(get_sr_for_name(&hir, "b"));
        assert_eq!(
            tt[&get_node_for_name(&hir, "o")],
            ConcretePacingType::Event(ActivationCondition::Conjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_trigger_simple() {
        let spec = "input a: Int8\n trigger a == 42";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        assert_eq!(
            tt[&NodeId::SRef(hir.triggers().nth(0).unwrap().sr)],
            ConcretePacingType::Event(ac_a)
        );
    }

    #[test]
    fn test_disjunction_annotated() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let ac_a = ActivationCondition::Stream(get_sr_for_name(&hir, "a"));
        let ac_b = ActivationCondition::Stream(get_sr_for_name(&hir, "b"));
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "x")],
            ConcretePacingType::Event(ActivationCondition::Disjunction(vec![ac_a, ac_b]))
        );
    }

    #[test]
    fn test_frequency_simple() {
        let spec = "output a: UInt8 @10Hz := 0";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        assert_eq!(
            tt[&get_node_for_name(&hir, "a")],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(10).unwrap()
            ))
        );
    }

    #[test]
    fn test_frequency_conjunction() {
        let spec = "output a: Int32 @10Hz := 0\noutput b: Int32 @5Hz := 0\noutput x := a+b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "x")],
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
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        //  input `a` has NodeId =  1
        let a_id = get_sr_for_name(&hir, "a");
        //  input `b` has NodeId =  3
        let b_id = get_sr_for_name(&hir, "b");
        //  input `c` has NodeId =  5
        let c_id = get_sr_for_name(&hir, "c");
        // output `x` has NodeId = 11
        let x_id = get_node_for_name(&hir, "x");
        // output `y` has NodeId = 19
        let y_id = get_node_for_name(&hir, "y");

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
    fn test_output_in_ac() {
        let spec = "input a: Int32\ninput b: Int32\ninput c: Int32\noutput x := a + b\noutput y @(x & c) := a + b + c";
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_counter() {
        let spec = "output b := b.offset(by: -1).defaults(to: 0) + 1";
        assert_eq!(1, num_errors(spec));
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
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(
                Rational::from_u8(1).unwrap()
            ))
        );
    }

    #[test]
    fn test_0_1hz_meet() {
        let spec =
            "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 0.3Hz := 1337\noutput c := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")],
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
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let i_type =
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")));
        assert_eq!(0, num_errors(spec));
        assert_eq!(tt[&get_node_for_name(&hir, "x")], i_type);
        assert_eq!(tt[&get_node_for_name(&hir, "y")], i_type);
    }

    #[test]
    fn test_parametric_output_parameter() {
        let spec = "input i: UInt8\noutput x(a: UInt8, b: Bool) @i := a";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "x")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\ninput i: UInt8\noutput o: Float32 @i := sin(2.0)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "o")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_tuple() {
        let spec = "input i: UInt8\noutput out: (Int8, Bool) @i:= (14, false)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "out")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")))
        );
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "out")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "in")))
        );
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(0, num_errors(spec));
        assert_eq!(
            tt[&get_node_for_name(&hir, "b")],
            ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "a")))
        );
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "out")],
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
