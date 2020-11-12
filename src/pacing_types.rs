use crate::rtltc::NodeId;
use biodivine_lib_bdd::boolean_expression::BooleanExpression;
use biodivine_lib_bdd::{Bdd, BddVariableSet};
use front::common_ir::StreamReference;
use front::hir::expression::{Constant, ConstantLiteral, Expression, ExpressionKind};
use front::parse::Span;
use front::reporting::{Handler, LabeledSpan};
use num::{CheckedDiv, Integer};
use rusttyc::{TcErr, TcKey};
use uom::lib::collections::HashMap;
use uom::num_rational::Ratio;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;

/// Parses either a periodic or an event based pacing type from an expression
pub fn parse_abstract_type(
    hir_expr: &Expression,
    var_set: &BddVariableSet,
    num_inputs: usize,
) -> Result<AbstractPacingType, String> {
    match &hir_expr.kind {
        ExpressionKind::LoadConstant(c) => match c {
            Constant::BasicConstant(cl) | Constant::InlinedConstant(cl, _) => match cl {
                ConstantLiteral::Bool(_) => {
                    let ac = parse_ac(hir_expr, var_set, num_inputs)?;
                    Ok(AbstractPacingType::Event(ac))
                }
                _ => Err("Cant infere pacing type of non bool constant".into()),
            },
        },
        _ => {
            let ac = parse_ac(hir_expr, var_set, num_inputs)?;
            Ok(AbstractPacingType::Event(ac))
        }
    }
}

fn parse_ac(
    ast_expr: &Expression,
    var_set: &BddVariableSet,
    num_inputs: usize,
) -> Result<Bdd, String> {
    use ExpressionKind::*;
    match &ast_expr.kind {
        LoadConstant(c) => match c {
            Constant::BasicConstant(cl) | Constant::InlinedConstant(cl, _) => match cl {
                ConstantLiteral::Bool(b) => {
                    if *b {
                        Ok(var_set.mk_true())
                    } else {
                        Err(
                            "Only 'True' is supported as literals in activation conditions.".into(), //l.span,
                        )
                    }
                }
                _ => Err("Only 'True' is supported as literals in activation conditions.".into()),
            },
        },
        StreamAccess(sref, kind, args) => {
            use front::hir::expression::StreamAccessKind;
            assert!(args.is_empty());
            assert!(matches!(kind, StreamAccessKind::Sync));
            let id = match sref {
                StreamReference::InRef(o) => *o,
                StreamReference::OutRef(o) => o + num_inputs,
            };
            Ok(var_set.mk_var_by_name(&id.to_string()))
        }
        /*
        Ident(i) => {
            let declartation = &decl[&ast_expr.id];
            let id = match declartation {
                Declaration::Out(out) => out.id,
                Declaration::ParamOut(param) => param.id,
                Declaration::In(input) => input.id,
                Declaration::Type(_)
                | Declaration::Param(_)
                | Declaration::Func(_)
                | Declaration::Const(_) => {
                    return Err((
                        "An activation condition can only refer to inputs or outputs.".into(),
                        i.span,
                    ));
                }
            };
            Ok(var_set.mk_var_by_name(&id.to_string()))
        }
        */
        ArithLog(op, v) => {
            assert!(
                v.len() == 2,
                "An activation condition can only contain literals and binary operators."
            );
            let ac_l = parse_ac(&v[0], var_set, num_inputs)?;
            let ac_r = parse_ac(&v[1], var_set, num_inputs)?;
            use front::hir::expression::ArithLogOp;
            match op {
                ArithLogOp::And => Ok(ac_l.and(&ac_r)),
                ArithLogOp::Or => Ok(ac_l.or(&ac_r)),
                _ => Err(
                    "Only '&' (and) or '|' (or) are allowed in activation conditions.".into(), //ast_expr.span,
                ),
            }
        }
        _ => Err(
            "An activation condition can only contain literals and binary operators.".into(), //ast_expr.span,
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    MixedEventPeriodic(AbstractPacingType, AbstractPacingType),
    //IncompatibleFrequencies(AbstractPacingType, AbstractPacingType),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PacingError {
    FreqAnnotationNeeded(Span),
    NeverEval(Span),
    MalformedAC(String),
}

impl PacingError {
    pub fn emit(self, handler: &Handler) {
        match self {
            PacingError::FreqAnnotationNeeded(span) => {
                let ls = LabeledSpan::new(span, "here", true);
                handler.error_with_span("Frequency annotation needed.", ls);
            }
            PacingError::NeverEval(span) => {
                let ls = LabeledSpan::new(span, "here", true);
                handler.error_with_span("The following stream is never evaluated.", ls);
            }
            PacingError::MalformedAC(reason) => {
                handler.error(&format!("Malformed activation condition: {}", reason));
            }
        }
    }

    pub fn emit_with_span(self, handler: &Handler, s: Span) {
        match self {
            PacingError::FreqAnnotationNeeded(_) | PacingError::NeverEval(_) => self.emit(handler),
            PacingError::MalformedAC(reason) => {
                let ls = LabeledSpan::new(s, "here", true);
                handler.error_with_span(&format!("Malformed activation condition: {}", reason), ls);
            }
        }
    }
}

fn bdd_to_string(bdd: &Bdd, vars: &BddVariableSet, input_names: &HashMap<NodeId, &str>) -> String {
    let exp = bdd.to_boolean_expression(vars);
    bexp_to_string(exp, input_names)
}
fn bexp_to_string(exp: BooleanExpression, input_names: &HashMap<NodeId, &str>) -> String {
    use BooleanExpression::*;
    match exp {
        Const(b) => {
            if b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Variable(s) => {
            /*
            let id = NodeId::new(s.parse::<usize>().unwrap());
            input_names[&id].to_string()
            */
            let idx = s.parse::<usize>().unwrap();
            let id = NodeId::SRef(StreamReference::InRef(idx));
            input_names[&id].to_string()
        }
        Not(exp) => format!("!{}", bexp_to_string(*exp, input_names)),
        And(l, r) => format!(
            "({} & {})",
            bexp_to_string(*l, input_names),
            bexp_to_string(*r, input_names)
        ),
        Or(l, r) => format!(
            "({} | {})",
            bexp_to_string(*l, input_names),
            bexp_to_string(*r, input_names)
        ),
        Xor(l, r) => format!(
            "({} ^ {})",
            bexp_to_string(*l, input_names),
            bexp_to_string(*r, input_names)
        ),
        Imp(l, r) => format!(
            "({} -> {})",
            bexp_to_string(*l, input_names),
            bexp_to_string(*r, input_names)
        ),
        Iff(l, r) => format!(
            "({} <-> {})",
            bexp_to_string(*l, input_names),
            bexp_to_string(*r, input_names)
        ),
    }
}

impl UnificationError {
    pub(crate) fn to_string(
        &self,
        bdd_vars: &BddVariableSet,
        input_name: &HashMap<NodeId, &str>,
    ) -> String {
        use UnificationError::*;
        match self {
            Other(s) => s.clone(),
            /*            IncompatibleFrequencies(f1, f2) => format!(
                "Found incompatible frequencies: '{}' and '{}'",
                f1.to_string(bdd_vars, input_name),
                f2.to_string(bdd_vars, input_name)
            ),*/
            MixedEventPeriodic(t1, t2) => format!(
                "Mixed event and periodic type: '{}' and '{}'",
                t1.to_string(bdd_vars, input_name),
                t2.to_string(bdd_vars, input_name)
            ),
        }
    }
}

pub(crate) fn emit_error(
    tce: &TcErr<AbstractPacingType>,
    handler: &Handler,
    vars: &BddVariableSet,
    spans: &HashMap<TcKey, Span>,
    names: &HashMap<NodeId, &str>,
) {
    match tce {
        TcErr::KeyEquation(k1, k2, err) => {
            let msg = &format!("In pacing type analysis:\n {}", err.to_string(vars, names));
            let span1 = LabeledSpan::new(spans[k1], "here", true);
            let span2 = LabeledSpan::new(spans[k2], "and here", true);
            let mut diag = handler.build_error_with_span(msg, span1);
            diag.add_labeled_span(span2);
            diag.emit();
        }
        TcErr::Bound(k1, k2o, err) => {
            let msg = &format!("In pacing type analysis:\n {}", err.to_string(vars, names));
            let span1 = LabeledSpan::new(spans[k1], "here", true);
            let mut diag = handler.build_error_with_span(msg, span1);
            if let Some(k2) = k2o {
                let span2 = LabeledSpan::new(spans[k2], "and here", false);
                diag.add_labeled_span(span2);
            }
            diag.emit();
        }
        TcErr::ChildAccessOutOfBound(key, ty, _idx) => {
            let msg = &format!(
                "In pacing type analysis:\n Child type out of bounds for type: {}",
                ty.to_string(vars, names)
            );
            let span = LabeledSpan::new(spans[key], "here", true);
            handler.error_with_span(msg, span);
        }
        TcErr::ExactTypeViolation(key, ty) => {
            let msg = &format!(
                "In pacing type analysis:\n Expected type: {}",
                ty.to_string(vars, names)
            );
            let span = LabeledSpan::new(spans[key], "here", true);
            handler.error_with_span(msg, span);
        }
        TcErr::ConflictingExactBounds(key, ty1, ty2) => {
            let msg = &format!(
                "In pacing type analysis:\n Conflicting type bounds: {} and {}",
                ty1.to_string(vars, names),
                ty2.to_string(vars, names)
            );
            let span = LabeledSpan::new(spans[key], "here", true);
            handler.error_with_span(msg, span);
        }
    }
}

// Abstract Type Definition

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Freq {
    Any,
    Fixed(UOM_Frequency),
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Freq::Any => write!(f, "Any"),
            Freq::Fixed(freq) => write!(
                f,
                "{}",
                freq.clone().into_format_args(
                    uom::si::frequency::hertz,
                    uom::fmt::DisplayStyle::Abbreviation
                )
            ),
        }
    }
}

impl Freq {
    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, UnificationError> {
        let lhs = match self {
            Freq::Fixed(f) => f,
            Freq::Any => return Ok(false),
        };
        let rhs = match other {
            Freq::Fixed(f) => f,
            Freq::Any => return Ok(false),
        };

        if lhs.get::<hertz>() < rhs.get::<hertz>() {
            return Ok(false);
        }
        match lhs.get::<hertz>().checked_div(&rhs.get::<hertz>()) {
            Some(q) => Ok(q.is_integer()),
            None => Err(UnificationError::Other(format!(
                "division of frequencies `{:?}`/`{:?}` failed",
                &lhs, &rhs
            ))),
        }
    }

    pub(crate) fn conjunction(&self, other: &Freq) -> Freq {
        let (numer_left, denom_left) = match self {
            Freq::Any => return *other,
            Freq::Fixed(f) => (*f.get::<hertz>().numer(), *f.get::<hertz>().denom()),
        };
        let (numer_right, denom_right) = match other {
            Freq::Any => return *self,
            Freq::Fixed(f) => (*f.get::<hertz>().numer(), *f.get::<hertz>().denom()),
        };
        // gcd(self, other) = gcd(numer_left, numer_right) / lcm(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `Rational`
        let r1: i64 = numer_left.gcd(&numer_right);
        let r2: i64 = denom_left.lcm(&denom_right);
        let r: Ratio<i64> = Ratio::new(r1, r2);
        Freq::Fixed(UOM_Frequency::new::<hertz>(r))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractPacingType {
    /// An event stream is extended when its activation condition is satisfied.
    Event(Bdd),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any,
    /// An event stream with this type is never evaluated.
    Never,
}

impl rusttyc::types::Abstract for AbstractPacingType {
    type Err = UnificationError;

    fn unconstrained() -> Self {
        AbstractPacingType::Any
    }

    fn meet(&self, other: &Self) -> Result<Self, Self::Err> {
        use AbstractPacingType::*;
        match (self, other) {
            (Any, x) | (x, Any) => Ok(x.clone()),
            (Never, x) | (x, Never) => Ok(x.clone()),
            (Event(ac), Periodic(f)) => Err(UnificationError::MixedEventPeriodic(
                Event(ac.clone()),
                Periodic(*f),
            )),
            (Periodic(f), Event(ac)) => Err(UnificationError::MixedEventPeriodic(
                Periodic(*f),
                Event(ac.clone()),
            )),
            (Event(ac1), Event(ac2)) => Ok(Event(ac1.and(&ac2))),
            (Periodic(f1), Periodic(f2)) => {
                if let Freq::Any = f1 {
                    Ok(Periodic(*f2))
                } else if let Freq::Any = f2 {
                    Ok(Periodic(*f1))
                } else {
                    Ok(Periodic(f1.conjunction(&f2)))
                }
            }
        }
    }

    fn arity(&self) -> Option<usize> {
        None
    }

    fn nth_child(&self, _n: usize) -> &Self {
        unreachable!()
    }

    fn with_children<I>(&self, _children: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        unreachable!()
    }
}

impl AbstractPacingType {
    pub(crate) fn to_string(&self, vars: &BddVariableSet, names: &HashMap<NodeId, &str>) -> String {
        match self {
            AbstractPacingType::Event(b) => format!("Event({})", bdd_to_string(b, vars, names)),
            AbstractPacingType::Periodic(freq) => format!("Periodic({})", freq),
            AbstractPacingType::Any => "Any".to_string(),
            AbstractPacingType::Never => "Never".to_string(),
        }
    }
}

// Concrete Type Definition

/**
The activation condition describes when an event-based stream produces a new value.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum ActivationCondition {
    /**
    When all of the activation conditions is true.
    */
    Conjunction(Vec<Self>),
    /**
    When one of the activation conditions is true.
    */
    Disjunction(Vec<Self>),
    /**
    Whenever the specified stream produces a new value.
    */
    Stream(NodeId),
    /**
    Whenever an event-based stream produces a new value.
    */
    True,
}

impl ActivationCondition {
    pub(crate) fn from_expression(exp: BooleanExpression) -> Result<Self, PacingError> {
        use BooleanExpression::*;
        match exp {
            Const(b) => {
                if b {
                    Ok(ActivationCondition::True)
                } else {
                    Err(PacingError::MalformedAC(
                        "False in Activation Condition".to_string(),
                    ))
                }
            }
            Variable(s) => {
                let id = s.parse::<usize>();
                match id {
                    Ok(i) => Ok(ActivationCondition::Stream(NodeId::SRef(
                        StreamReference::InRef(i),
                    ))),
                    Err(_) => Err(PacingError::MalformedAC("Wrong Variable in AC".to_string())),
                }
            }
            And(left, right) => {
                let l = ActivationCondition::from_expression(*left)?;
                let r = ActivationCondition::from_expression(*right)?;
                match (l, r) {
                    (
                        ActivationCondition::Conjunction(mut left),
                        ActivationCondition::Conjunction(mut right),
                    ) => {
                        left.append(&mut right);
                        left.dedup();
                        left.sort();
                        Ok(ActivationCondition::Conjunction(left))
                    }
                    (ActivationCondition::Conjunction(mut other_con), other_ac)
                    | (other_ac, ActivationCondition::Conjunction(mut other_con)) => {
                        other_con.push(other_ac);
                        other_con.dedup();
                        other_con.sort();
                        Ok(ActivationCondition::Conjunction(other_con))
                    }
                    (a, b) => Ok(ActivationCondition::Conjunction(vec![a, b])),
                }
            }
            Or(left, right) => {
                let l = ActivationCondition::from_expression(*left)?;
                let r = ActivationCondition::from_expression(*right)?;
                match (l, r) {
                    (
                        ActivationCondition::Disjunction(mut left),
                        ActivationCondition::Disjunction(mut right),
                    ) => {
                        left.append(&mut right);
                        left.dedup();
                        left.sort();
                        Ok(ActivationCondition::Disjunction(left))
                    }
                    (ActivationCondition::Disjunction(mut other_con), other_ac)
                    | (other_ac, ActivationCondition::Disjunction(mut other_con)) => {
                        other_con.push(other_ac);
                        other_con.dedup();
                        other_con.sort();
                        Ok(ActivationCondition::Disjunction(other_con))
                    }
                    (a, b) => Ok(ActivationCondition::Disjunction(vec![a, b])),
                }
            }
            _ => Err(PacingError::MalformedAC(
                "Unsupported Operation in AC".to_string(),
            )),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum ConcretePacingType {
    /// The stream / expression can be evaluated whenever the activation condition is satisfied.
    Event(ActivationCondition),
    /// The stream / expression can be evaluated with a fixed frequency.
    FixedPeriodic(UOM_Frequency),
    /// The stream / expression can be evaluated with any frequency.
    Periodic,
    /// The stream / expression can always be evaluated.
    Constant,
}

impl ConcretePacingType {
    pub(crate) fn from_abstract(
        abs_t: AbstractPacingType,
        vars: &BddVariableSet,
    ) -> Result<Self, PacingError> {
        match abs_t {
            AbstractPacingType::Any => Ok(ConcretePacingType::Constant),
            AbstractPacingType::Event(b) => {
                ActivationCondition::from_expression(b.to_boolean_expression(vars))
                    .map(ConcretePacingType::Event)
            }
            AbstractPacingType::Periodic(freq) => match freq {
                Freq::Fixed(f) => Ok(ConcretePacingType::FixedPeriodic(f)),
                Freq::Any => Ok(ConcretePacingType::Periodic),
            },
            AbstractPacingType::Never => unreachable!(),
        }
    }

    pub(crate) fn to_abstract_freq(&self) -> Result<AbstractPacingType, String> {
        match self {
            ConcretePacingType::FixedPeriodic(f) => {
                Ok(AbstractPacingType::Periodic(Freq::Fixed(*f)))
            }
            ConcretePacingType::Periodic => Ok(AbstractPacingType::Periodic(Freq::Any)),
            _ => Err("Supplied invalid concrete pacing type.".to_string()),
        }
    }
}
