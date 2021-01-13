use super::*;
extern crate regex;

use crate::tyc::pacing_types::{
    AbstractExpressionType, AbstractPacingType, ActivationCondition, Freq, PacingError, PacingErrorKind, StreamTypeKeys,
};

use crate::common_ir::Offset;
use crate::hir::expression::{Expression, ExpressionKind};
use crate::hir::modes::ir_expr::WithIrExpr;
use crate::hir::modes::HirMode;
use crate::hir::{Input, Output, Trigger, AC};
use crate::reporting::Span;
use crate::tyc::rtltc::NodeId;
use crate::RTLolaHIR;
use rusttyc::types::AbstractTypeTable;
use rusttyc::{TcKey, TypeChecker};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable(String);

impl rusttyc::TcVar for Variable {}

pub struct Context<'a, M>
where
    M: HirMode + WithIrExpr + 'static,
{
    pub(crate) hir: &'a RTLolaHIR<M>,
    pub(crate) pacing_tyc: TypeChecker<AbstractPacingType, Variable>,
    pub(crate) expression_tyc: TypeChecker<AbstractExpressionType, Variable>,
    pub(crate) node_key: HashMap<NodeId, StreamTypeKeys>,
    pub(crate) pacing_key_span: HashMap<TcKey, Span>,
    pub(crate) expression_key_span: HashMap<TcKey, Span>,
}

impl<'a, M> Context<'a, M>
where
    M: HirMode + WithIrExpr + 'static,
{
    pub(crate) fn new(hir: &'a RTLolaHIR<M>) -> Self {
        let node_key = HashMap::new();
        let pacing_tyc = TypeChecker::new();
        let expression_tyc = TypeChecker::new();
        let pacing_key_span = HashMap::new();
        let expression_key_span = HashMap::new();

        let mut res = Context { hir, pacing_tyc, expression_tyc, node_key, pacing_key_span, expression_key_span };
        res.init();
        res
    }

    pub(crate) fn new_stream_key(&mut self) -> StreamTypeKeys {
        StreamTypeKeys {
            exp_pacing: self.pacing_tyc.new_term_key(),
            spawn: (self.pacing_tyc.new_term_key(), self.expression_tyc.new_term_key()),
            filter: self.expression_tyc.new_term_key(),
            close: self.expression_tyc.new_term_key(),
        }
    }

    pub(crate) fn add_stream_key_span(&mut self, keys: StreamTypeKeys, span: Span) {
        self.pacing_key_span.insert(keys.exp_pacing, span.clone());
        self.pacing_key_span.insert(keys.spawn.0, span.clone());
        self.expression_key_span.insert(keys.spawn.1, span.clone());
        self.expression_key_span.insert(keys.filter, span.clone());
        self.expression_key_span.insert(keys.close, span);
    }

    pub(crate) fn init(&mut self) {
        for input in self.hir.inputs() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(input.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, input.span.clone());
            self.pacing_key_span.insert(key.spawn.0, Span::Unknown);
            self.expression_key_span.insert(key.spawn.1, Span::Unknown);
            self.expression_key_span.insert(key.filter, Span::Unknown);
            self.expression_key_span.insert(key.close, Span::Unknown);
        }
        for output in self.hir.outputs() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(output.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, output.span.clone());
            self.pacing_key_span.insert(
                key.spawn.0,
                output
                    .instance_template
                    .spawn
                    .and_then(|spawn| spawn.condition)
                    .map(|id| self.hir.expression(id).span.clone())
                    .unwrap_or(Span::Unknown),
            );
            self.pacing_key_span.insert(
                key.spawn.1,
                output
                    .instance_template
                    .spawn
                    .and_then(|spawn| spawn.target)
                    .map(|id| self.hir.expression(id).span.clone())
                    .unwrap_or(Span::Unknown),
            );
            self.pacing_key_span.insert(
                key.filter,
                output.instance_template.filter.map(|id| self.hir.expression(id).span.clone()).unwrap_or(Span::Unknown),
            );
            self.pacing_key_span.insert(
                key.close,
                output.instance_template.filter.map(|id| self.hir.expression(id).span.clone()).unwrap_or(Span::Unknown),
            );
        }

        for trigger in self.hir.triggers() {
            let key = self.new_stream_key();
            self.node_key.insert(NodeId::SRef(trigger.sr), key);
            self.pacing_key_span.insert(key.exp_pacing, trigger.span.clone());
            self.pacing_key_span.insert(key.spawn.0, Span::Unknown);
            self.expression_key_span.insert(key.spawn.1, Span::Unknown);
            self.expression_key_span.insert(key.filter, Span::Unknown);
            self.expression_key_span.insert(key.close, Span::Unknown);
        }
    }

    pub(crate) fn input_infer(&mut self, input: &Input) -> Result<(), PacingError> {
        let ac = AbstractPacingType::Event(ActivationCondition::Stream(input.sr));
        let input_key = self.node_key[&NodeId::SRef(input.sr)].exp_pacing;
        self.pacing_tyc.impose(input_key.concretizes_explicit(ac)).map_err(PacingError::from)
    }

    pub(crate) fn trigger_infer(&mut self, trigger: &Trigger) -> Result<(), PacingError> {
        let ex_key = self.expression_infer(self.hir.expr(trigger.sr))?;
        let trigger_key = self.node_key[&NodeId::SRef(trigger.sr)].exp_pacing;
        self.pacing_tyc.impose(trigger_key.equate_with(ex_key.exp_pacing)).map_err(PacingError::from)
    }

    pub(crate) fn output_infer(&mut self, output: &Output) -> Result<(), PacingError> {
        let stream_keys = self.node_key[&NodeId::SRef(output.sr)];

        // Type Expression Pacing
        let exp_key = self.expression_infer(&self.hir.expr(output.sr))?;
        self.pacing_tyc.impose(stream_keys.exp_pacing.concretizes_explicit(AbstractPacingType::Never))?;

        // Check if there is a type is annotated
        if let Some(ac) = &output.activation_condition {
            let annotated_ty_key = self.pacing_tyc.new_term_key();
            let annotated_ty = match ac {
                AC::Frequency { span, value } => {
                    self.pacing_key_span.insert(annotated_ty_key, span.clone());
                    AbstractPacingType::Periodic(Freq::Fixed(*value))
                }
                AC::Expr(eid) => {
                    let expr = self.hir.expression(*eid);
                    self.pacing_key_span.insert(annotated_ty_key, expr.span.clone());
                    AbstractPacingType::Event(ActivationCondition::parse(expr).map_err(|e| PacingError {
                        kind: e.kind,
                        key1: Some(annotated_ty_key),
                        key2: None,
                    })?)
                }
            };
            // Bind key to parsed type
            self.pacing_tyc.impose(annotated_ty_key.has_exactly_type(annotated_ty))?;

            // Annotated type should be more concrete than inferred type
            self.pacing_tyc.impose(annotated_ty_key.concretizes(exp_key.exp_pacing))?;

            // Output type is equal to declared type
            self.pacing_tyc.impose(stream_keys.exp_pacing.equate_with(annotated_ty_key)).map_err(PacingError::from)
        } else {
            // Output type is equal to inferred type
            self.pacing_tyc.impose(stream_keys.exp_pacing.concretizes(exp_key.exp_pacing)).map_err(PacingError::from)
        }
    }

    pub(crate) fn expression_infer(&mut self, exp: &Expression) -> Result<StreamTypeKeys, PacingError> {
        let term_keys: StreamTypeKeys = self.new_stream_key();
        use AbstractPacingType::*;
        match &exp.kind {
            ExpressionKind::LoadConstant(_) | ExpressionKind::ParameterAccess(_, _) => {
                //constants have arbitrary pacing type
                self.pacing_tyc.impose(term_keys.exp_pacing.has_exactly_type(Any))?;
            }
            ExpressionKind::StreamAccess(sref, kind, args) => {
                use crate::common_ir::StreamAccessKind;
                let stream_key = self.node_key[&NodeId::SRef(*sref)];
                match kind {
                    StreamAccessKind::Sync => {
                        self.pacing_tyc.impose(term_keys.exp_pacing.equate_with(stream_key.exp_pacing))?
                    }
                    StreamAccessKind::Offset(off) => {
                        match off {
                            Offset::PastRealTimeOffset(_) | Offset::FutureRealTimeOffset(_) => {
                                // Real time offset are only allowed on timed streams.
                                self.pacing_tyc
                                    .impose(term_keys.exp_pacing.concretizes_explicit(Periodic(Freq::Any)))?;
                                self.pacing_tyc.impose(term_keys.exp_pacing.concretizes(stream_key.exp_pacing))?;
                            }
                            Offset::PastDiscreteOffset(_) | Offset::FutureDiscreteOffset(_) => {
                                self.pacing_tyc.impose(term_keys.exp_pacing.concretizes(stream_key.exp_pacing))?;
                            }
                        }
                    }
                    StreamAccessKind::Hold => self.pacing_tyc.impose(term_keys.exp_pacing.concretizes_explicit(Any))?,
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => {
                        self.pacing_tyc.impose(term_keys.exp_pacing.concretizes_explicit(Periodic(Freq::Any)))?;
                        // Not needed as the pacing of a sliding window is only bound to the frequency of the stream it is contained in.
                    }
                };

                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.pacing_tyc.impose(term_keys.exp_pacing.concretizes(arg_key.exp_pacing))?;
                }
            }
            ExpressionKind::Default { expr, default } => {
                let ex_key = self.expression_infer(&*expr)?;
                let def_key = self.expression_infer(&*default)?;

                self.pacing_tyc.impose(term_keys.exp_pacing.is_meet_of(ex_key.exp_pacing, def_key.exp_pacing))?;
            }
            ExpressionKind::ArithLog(_, args) => match args.len() {
                2 => {
                    let left_key = self.expression_infer(&args[0])?;
                    let right_key = self.expression_infer(&args[1])?;

                    self.pacing_tyc
                        .impose(term_keys.exp_pacing.is_meet_of(left_key.exp_pacing, right_key.exp_pacing))?;
                }
                1 => {
                    let ex_key = self.expression_infer(&args[0])?;

                    self.pacing_tyc.impose(term_keys.exp_pacing.equate_with(ex_key.exp_pacing))?;
                }
                _ => unreachable!(),
            },
            ExpressionKind::Ite { condition, consequence, alternative } => {
                let cond_key = self.expression_infer(&*condition)?;
                let cons_key = self.expression_infer(&*consequence)?;
                let alt_key = self.expression_infer(&*alternative)?;

                self.pacing_tyc.impose(term_keys.exp_pacing.is_meet_of_all(&[
                    cond_key.exp_pacing,
                    cons_key.exp_pacing,
                    alt_key.exp_pacing,
                ]))?;
            }
            ExpressionKind::Tuple(elements) => {
                let ele_keys: Vec<TcKey> = elements
                    .iter()
                    .map(|e| self.expression_infer(&*e).map(|keys| keys.exp_pacing))
                    .collect::<Result<Vec<TcKey>, PacingError>>()?;
                self.pacing_tyc.impose(term_keys.exp_pacing.is_meet_of_all(&ele_keys))?;
            }
            ExpressionKind::Function { args, .. } => {
                for arg in args {
                    let arg_key = self.expression_infer(&*arg)?;
                    self.pacing_tyc.impose(term_keys.exp_pacing.concretizes(arg_key.exp_pacing))?;
                }
            }
            ExpressionKind::TupleAccess(t, _) => {
                let exp_key = self.expression_infer(&*t)?;
                self.pacing_tyc.impose(term_keys.exp_pacing.equate_with(exp_key.exp_pacing))?;
            }
            ExpressionKind::Widen(inner, _) => {
                let exp_key = self.expression_infer(&*inner)?;
                self.pacing_tyc.impose(term_keys.exp_pacing.equate_with(exp_key.exp_pacing))?;
            }
        };
        self.node_key.insert(NodeId::Expr(exp.eid), term_keys);
        self.add_stream_key_span(term_keys, exp.span.clone());
        Ok(term_keys)
    }

    pub(crate) fn post_process(
        hir: &RTLolaHIR<M>,
        nid_key: HashMap<NodeId, StreamTypeKeys>,
        tt: &AbstractTypeTable<AbstractPacingType>,
    ) -> Vec<PacingError> {
        let mut res = vec![];
        // That every periodic stream has a frequency
        for output in hir.outputs() {
            let at = &tt[nid_key[&NodeId::SRef(output.sr)].exp_pacing];
            match at {
                AbstractPacingType::Periodic(Freq::Any) => {
                    res.push(PacingErrorKind::FreqAnnotationNeeded(output.span.clone()).into());
                }
                AbstractPacingType::Never => {
                    res.push(PacingErrorKind::NeverEval(output.span.clone()).into());
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
    use crate::common_ir::StreamReference;
    use crate::hir::modes::IrExpression;
    use crate::hir::RTLolaHIR;
    use crate::reporting::Handler;
    use crate::tyc::pacing_types::{ActivationCondition, ConcretePacingType};
    use crate::tyc::rtltc::NodeId;
    use crate::tyc::LolaTypeChecker;
    use crate::RTLolaAst;
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use std::path::PathBuf;
    use uom::si::frequency::hertz;
    use uom::si::rational64::Frequency as UOM_Frequency;

    fn setup_ast(spec: &str) -> (RTLolaHIR<IrExpression>, Handler) {
        let handler = crate::reporting::Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst = match crate::parse::parse(spec, &handler, crate::FrontendConfig::default()) {
            Ok(s) => s,
            Err(e) => panic!("Spech {} cannot be parsed: {}", spec, e),
        };
        let hir = crate::hir::RTLolaHIR::<IrExpression>::from_ast(ast, &handler, &crate::FrontendConfig::default());
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
        assert_eq!(tt[&NodeId::SRef(hir.triggers().nth(0).unwrap().sr)], ConcretePacingType::Event(ac_a));
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
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(10).unwrap()))
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
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap()))
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
        let spec = "output a @4Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // equivalent to b[-500ms].defaults(to: 0) + a[-250ms].defaults(to: 0)
        assert_eq!(0, num_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_incompatible() {
        let spec = "output a @3Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(1, num_errors(spec));
    }

    #[test]
    fn test_simple_loop() {
        let spec = "output a @1Hz := a";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(0, num_errors(spec));
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
        let spec = "input i: Int64\noutput a @ 5Hz := 42\noutput b @ 2Hz := 1337\noutput c := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(1).unwrap()))
        );
    }

    #[test]
    fn test_0_1hz_meet() {
        let spec = "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 0.3Hz := 1337\noutput c := a + b";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        assert_eq!(num_errors(spec), 0);

        assert_eq!(
            tt[&get_node_for_name(&hir, "c")],
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_f32(0.1).unwrap()))
        );
    }

    #[test]
    fn test_annotated_freq() {
        let spec = "input i: Int64\noutput a @ 2Hz := 42\noutput b @ 3Hz := 1337\noutput c @2Hz := a + b";
        assert_eq!(num_errors(spec), 1);
    }

    #[test]
    fn test_parametric_output() {
        let spec = "input i: UInt8\noutput x(a: UInt8, b: Bool): Int8 := i\noutput y := x(1, false)";
        let (hir, handler) = setup_ast(spec);
        let mut ltc = LolaTypeChecker::new(&hir, &handler);
        let tt = ltc.pacing_type_infer().unwrap();
        let i_type = ConcretePacingType::Event(ActivationCondition::Stream(get_sr_for_name(&hir, "i")));
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
            ConcretePacingType::FixedPeriodic(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap()))
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
