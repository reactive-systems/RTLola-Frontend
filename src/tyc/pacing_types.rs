use crate::common_ir::{StreamAccessKind, StreamReference};
use crate::hir::expression::{Constant, ConstantLiteral, Expression, ExpressionKind};
use crate::reporting::{Diagnostic, Handler, Span};
use itertools::Itertools;
use num::{CheckedDiv, Integer};
use rusttyc::{TcErr, TcKey};
use uom::lib::collections::HashMap;
use uom::lib::fmt::Formatter;
use uom::num_rational::Ratio;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;

/// The activation condition describes when an event-based stream produces a new value.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Ord, Hash)]
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
    Stream(StreamReference),
    /**
    Whenever an event-based stream produces a new value.
    */
    True,
}

/// Represents the frequency of a periodic stream
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) enum Freq {
    Any,
    Fixed(UOM_Frequency),
}

/// The internal representation of a pacing type
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AbstractPacingType {
    /// An event stream is extended when its activation condition is satisfied.
    Event(ActivationCondition),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any,
    /// An event stream with this type is never evaluated.
    Never,
}

/// The internal representation of an expression type
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AbstractExpressionType {
    Any,
    Expression(Expression),
}

/// The internal representation of the overall Stream pacing
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AbstractStreamPacing {
    pub exp_pacing: AbstractPacingType,
    pub spawn: (AbstractPacingType, AbstractExpressionType),
    pub filter: AbstractExpressionType,
    pub close: AbstractExpressionType,
}

/// The general error structure
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PacingError {
    FreqAnnotationNeeded(Span),
    NeverEval(Span),
    MalformedAC(Span, String),
    MixedEventPeriodic(AbstractPacingType, AbstractPacingType),
    IncompatibleExpressions(AbstractExpressionType, AbstractExpressionType),
    ParameterizedExpr(Span),
    Other(Span, String),
}

/// The external definition of a pacing type
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

impl ActivationCondition {
    pub(crate) fn and(self, other: Self) -> Self {
        let ac = match (self, other) {
            (ActivationCondition::Conjunction(mut left), ActivationCondition::Conjunction(mut right)) => {
                left.append(&mut right);
                left.sort();
                left.dedup();
                ActivationCondition::Conjunction(left)
            }
            (ActivationCondition::True, other) | (other, ActivationCondition::True) => other,
            (ActivationCondition::Conjunction(mut other_con), other_ac)
            | (other_ac, ActivationCondition::Conjunction(mut other_con)) => {
                other_con.push(other_ac);
                other_con.sort();
                other_con.dedup();
                ActivationCondition::Conjunction(other_con)
            }
            (a, b) => {
                let mut childs = vec![a, b];
                childs.sort();
                childs.dedup();
                ActivationCondition::Conjunction(childs)
            }
        };
        match &ac {
            ActivationCondition::Conjunction(v) | ActivationCondition::Disjunction(v) => {
                if v.len() == 1 {
                    return v[0].clone();
                }
            }
            _ => {}
        }
        ac
    }
    pub(crate) fn or(self, other: Self) -> Self {
        let ac = match (self, other) {
            (ActivationCondition::Disjunction(mut left), ActivationCondition::Disjunction(mut right)) => {
                left.append(&mut right);
                left.sort();
                left.dedup();
                ActivationCondition::Disjunction(left)
            }
            (ActivationCondition::True, _) | (_, ActivationCondition::True) => ActivationCondition::True,
            (ActivationCondition::Disjunction(mut other_dis), other_ac)
            | (other_ac, ActivationCondition::Disjunction(mut other_dis)) => {
                other_dis.push(other_ac);
                other_dis.sort();
                other_dis.dedup();
                ActivationCondition::Disjunction(other_dis)
            }
            (a, b) => {
                let mut childs = vec![a, b];
                childs.sort();
                childs.dedup();
                ActivationCondition::Disjunction(childs)
            }
        };
        match &ac {
            ActivationCondition::Conjunction(v) | ActivationCondition::Disjunction(v) => {
                if v.len() == 1 {
                    return v[0].clone();
                }
            }
            _ => {}
        }
        ac
    }

    pub(crate) fn parse(ast_expr: &Expression) -> Result<Self, PacingError> {
        use ExpressionKind::*;
        match &ast_expr.kind {
            LoadConstant(c) => match c {
                Constant::BasicConstant(cl) | Constant::InlinedConstant(cl, _) => match cl {
                    ConstantLiteral::Bool(b) => {
                        if *b {
                            Ok(ActivationCondition::True)
                        } else {
                            Err(PacingError::MalformedAC(
                                ast_expr.span.clone(),
                                "Only 'True' is supported as literals in activation conditions.".into(),
                            ))
                        }
                    }
                    _ => Err(PacingError::MalformedAC(
                        ast_expr.span.clone(),
                        "Only 'True' is supported as literals in activation conditions.".into(),
                    )),
                },
            },
            StreamAccess(sref, kind, args) => {
                if !args.is_empty() {
                    return Err(PacingError::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    ));
                }
                match kind {
                    StreamAccessKind::Sync => {}
                    _ => {
                        return Err(PacingError::MalformedAC(
                            ast_expr.span.clone(),
                            "An activation condition can only contain literals and binary operators.".into(),
                        ));
                    }
                }
                if sref.is_output() {
                    return Err(PacingError::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only refer to input streams".into(),
                    ));
                }
                Ok(ActivationCondition::Stream(*sref))
            }
            ArithLog(op, v) => {
                if v.len() != 2 {
                    return Err(PacingError::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    ));
                }
                let ac_l = Self::parse(&v[0])?;
                let ac_r = Self::parse(&v[1])?;
                use crate::hir::expression::ArithLogOp;
                match op {
                    ArithLogOp::And | ArithLogOp::BitAnd => Ok(ac_l.and(ac_r)),
                    ArithLogOp::Or | ArithLogOp::BitOr => Ok(ac_l.or(ac_r)),
                    _ => Err(PacingError::MalformedAC(
                        ast_expr.span.clone(),
                        "Only '&' (and) or '|' (or) are allowed in activation conditions.".into(),
                    )),
                }
            }
            _ => Err(PacingError::MalformedAC(
                ast_expr.span.clone(),
                "An activation condition can only contain literals and binary operators.".into(),
            )),
        }
    }

    pub fn to_string(&self, stream_names: &HashMap<StreamReference, &str>) -> String {
        use ActivationCondition::*;
        match self {
            True => "⊤".into(),
            Stream(sr) => stream_names[&sr].into(),
            Conjunction(childs) => {
                let child_string: String = childs.iter().map(|ac| ac.to_string(stream_names)).join(" ∧ ");
                format!("({})", child_string)
            }
            Disjunction(childs) => {
                let child_string: String = childs.iter().map(|ac| ac.to_string(stream_names)).join(" ∨ ");
                format!("({})", child_string)
            }
        }
    }
}

impl PacingError {
    pub fn emit(
        &self,
        handler: &Handler,
        spans: &HashMap<TcKey, Span>,
        names: &HashMap<StreamReference, &str>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    ) {
        match self {
            PacingError::FreqAnnotationNeeded(span) => {
                handler.error_with_span(
                    "In pacing type analysis:\nFrequency annotation needed.",
                    span.clone(),
                    Some("here"),
                );
            }
            PacingError::NeverEval(span) => {
                Diagnostic::error(handler, "In pacing type analysis:\nThe following stream is never evaluated.")
                    .add_span_with_label(span.clone(), Some("here"), true)
                    .add_note("Help: Consider annotating a pacing type explicitly.")
                    .emit();
            }
            PacingError::MalformedAC(span, reason) => {
                handler.error_with_span(
                    &format!("In pacing type analysis:\nMalformed activation condition: {}", reason),
                    span.clone(),
                    Some("here"),
                );
            }
            PacingError::MixedEventPeriodic(absty1, absty2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                let ty1 = absty1.to_string(names);
                let ty2 = absty2.to_string(names);
                Diagnostic::error(
                    handler,
                    format!("In pacing type analysis:\nMixed an event and a periodic type: {} and {}", ty1, ty2)
                        .as_str(),
                )
                .maybe_add_span_with_label(span1, Some(format!("Found {} here", ty1).as_str()), true)
                .maybe_add_span_with_label(span2, Some(format!("and found {} here", ty2).as_str()), false)
                .emit();
            }
            PacingError::IncompatibleExpressions(e1, e2) => {
                let span1 = key1.and_then(|k| spans.get(&k).cloned());
                let span2 = key2.and_then(|k| spans.get(&k).cloned());
                Diagnostic::error(
                    handler,
                    format!("In pacing type analysis:\nIncompatible expressions: {} and {}", e1, e2).as_str(),
                )
                .maybe_add_span_with_label(span1, Some(format!("Found {} here", e1).as_str()), true)
                .maybe_add_span_with_label(span2, Some(format!("and found {} here", e2).as_str()), false)
                .emit();
            }
            PacingError::Other(span, reason) => {
                handler.error_with_span(
                    format!("In pacing type analysis:\n{}", reason).as_str(),
                    span.clone(),
                    Some("here"),
                );
            }
            PacingError::ParameterizedExpr(span) => {
                Diagnostic::error(
                    handler,
                    "In pacing type analysis:\nExpression of stream 'a' accesses a parameterized stream 'b', but stream 'a' is not parameterized."
                )
                    .add_span_with_label(span.clone(), Some("here"), true)
                    .add_note("Help: Consider using a hold access")
                    .emit();
            }
        }
    }
}

pub(crate) fn emit_error(
    tce: &TcErr<AbstractPacingType>,
    handler: &Handler,
    spans: &HashMap<TcKey, Span>,
    names: &HashMap<StreamReference, &str>,
) {
    match tce {
        TcErr::KeyEquation(k1, k2, err) => {
            err.emit(handler, spans, names, Some(*k1), Some(*k2));
        }
        TcErr::Bound(k1, k2, err) => {
            err.emit(handler, spans, names, Some(*k1), *k2);
        }
        TcErr::ChildAccessOutOfBound(key, ty, _idx) => {
            let msg = &format!("In pacing type analysis:\n Child type out of bounds for type: {}", ty.to_string(names));
            handler.error_with_span(msg, spans.get(key).unwrap_or(&Span::Unknown).clone(), Some("here"));
        }
        TcErr::ExactTypeViolation(key, ty) => {
            let msg = &format!("In pacing type analysis:\n Expected type: {}", ty.to_string(names));
            handler.error_with_span(msg, spans.get(key).unwrap_or(&Span::Unknown).clone(), Some("here"));
        }
        TcErr::ConflictingExactBounds(key, ty1, ty2) => {
            let msg = &format!(
                "In pacing type analysis:\n Conflicting type bounds: {} and {}",
                ty1.to_string(names),
                ty2.to_string(names)
            );
            handler.error_with_span(msg, spans.get(key).unwrap_or(&Span::Unknown).clone(), Some("here"));
        }
    }
}

// Abstract Type Definition

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Freq::Any => write!(f, "Any"),
            Freq::Fixed(freq) => {
                write!(f, "{}", freq.clone().into_format_args(hertz, uom::fmt::DisplayStyle::Abbreviation))
            }
        }
    }
}

impl Freq {
    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, PacingError> {
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
            None => Err(PacingError::Other(
                Span::Unknown,
                format!("division of frequencies `{:?}`/`{:?}` failed", &lhs, &rhs),
            )),
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

impl rusttyc::types::Abstract for AbstractPacingType {
    type Err = PacingError;

    fn unconstrained() -> Self {
        AbstractPacingType::Any
    }

    fn meet(&self, other: &Self) -> Result<Self, Self::Err> {
        use AbstractPacingType::*;
        match (self, other) {
            (Any, x) | (x, Any) => Ok(x.clone()),
            (Never, x) | (x, Never) => Ok(x.clone()),
            (Event(ac), Periodic(f)) => Err(PacingError::MixedEventPeriodic(Event(ac.clone()), Periodic(*f))),
            (Periodic(f), Event(ac)) => Err(PacingError::MixedEventPeriodic(Periodic(*f), Event(ac.clone()))),
            (Event(ac1), Event(ac2)) => Ok(Event(ac1.clone().and(ac2.clone()))),
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
    pub(crate) fn to_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            AbstractPacingType::Event(ac) => format!("Event({})", ac.to_string(names)),
            AbstractPacingType::Periodic(freq) => format!("Periodic({})", freq),
            AbstractPacingType::Any => "Any".to_string(),
            AbstractPacingType::Never => "Never".to_string(),
        }
    }
}

impl rusttyc::types::Abstract for AbstractExpressionType {
    type Err = PacingError;

    fn unconstrained() -> Self {
        Self::Any
    }

    fn meet(&self, other: &Self) -> Result<Self, Self::Err> {
        match (self, other) {
            (Self::Any, x) | (x, Self::Any) => Ok(x.clone()),
            (Self::Expression(a), Self::Expression(b)) => {
                if a == b {
                    Ok(Self::Expression(a.clone()))
                } else {
                    Err(PacingError::IncompatibleExpressions(Self::Expression(a.clone()), Self::Expression(b.clone())))
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

impl std::fmt::Display for AbstractExpressionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "Any"),
            Self::Expression(e) => write!(f, "Exp({})", e),
        }
    }
}

impl rusttyc::types::Abstract for AbstractStreamPacing {
    type Err = PacingError;

    fn unconstrained() -> Self {
        Self::top()
    }

    fn meet(&self, other: &Self) -> Result<Self, Self::Err> {
        let AbstractStreamPacing { exp_pacing: exp1, spawn: s1, filter: f1, close: c1 } = self;
        let AbstractStreamPacing { exp_pacing: exp2, spawn: s2, filter: f2, close: c2 } = other;
        Ok(AbstractStreamPacing {
            exp_pacing: exp1.meet(exp2)?,
            spawn: (s1.0.meet(&s2.0)?, s1.1.meet(&s2.1)?),
            filter: f1.meet(f2)?,
            close: c1.meet(c2)?,
        })
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

impl AbstractStreamPacing {
    pub fn top() -> Self {
        AbstractStreamPacing {
            exp_pacing: AbstractPacingType::Any,
            spawn: (AbstractPacingType::Any, AbstractExpressionType::Any),
            filter: AbstractExpressionType::Any,
            close: AbstractExpressionType::Any,
        }
    }

    pub fn top_with_pacing(pacing: AbstractPacingType) -> Self {
        let mut res = Self::top();
        res.exp_pacing = pacing;
        res
    }

    pub(crate) fn to_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        format!(
            "(exp pacing: {}, spawn: ({}, {}), filter: {}, close: {})",
            self.exp_pacing.to_string(names),
            self.spawn.0.to_string(names),
            self.spawn.1,
            self.filter,
            self.close
        )
    }
}

impl ConcretePacingType {
    pub(crate) fn from_abstract(abs_t: AbstractPacingType) -> Result<Self, PacingError> {
        match abs_t {
            AbstractPacingType::Any => Ok(ConcretePacingType::Constant),
            AbstractPacingType::Event(ac) => Ok(ConcretePacingType::Event(ac)),
            AbstractPacingType::Periodic(freq) => match freq {
                Freq::Fixed(f) => Ok(ConcretePacingType::FixedPeriodic(f)),
                Freq::Any => Ok(ConcretePacingType::Periodic),
            },
            AbstractPacingType::Never => {
                Err(PacingError::Other(Span::Unknown, "Tried to concretize abstract type never!".into()))
            }
        }
    }

    pub(crate) fn to_abstract_freq(&self) -> Result<AbstractPacingType, String> {
        match self {
            ConcretePacingType::FixedPeriodic(f) => Ok(AbstractPacingType::Periodic(Freq::Fixed(*f))),
            ConcretePacingType::Periodic => Ok(AbstractPacingType::Periodic(Freq::Any)),
            _ => Err("Supplied invalid concrete pacing type.".to_string()),
        }
    }
}
