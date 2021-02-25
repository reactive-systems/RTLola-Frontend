use super::rusttyc::Partial;
use super::rusttyc::{Constructable, Variant};
use crate::common_ir::{StreamAccessKind, StreamReference};
use crate::hir::expression::{Constant, ConstantLiteral, ExprId, Expression, ExpressionKind};
use crate::hir::modes::ir_expr::WithIrExpr;
use crate::hir::modes::HirMode;
use crate::hir::AC;
use crate::reporting::{Diagnostic, Handler, Span};
use crate::RTLolaHIR;
use itertools::Itertools;
use num::{CheckedDiv, Integer};
use rusttyc::{TcErr, TcKey};
use std::fmt::Debug;
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
}

/// The internal representation of an expression type
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AbstractExpressionType {
    Any,
    AnyClose,
    Expression(Expression),
}

/// The internal representation of the overall Stream pacing
/// Types are given by keys in the respective rustic instances
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamTypeKeys {
    /// Key to the AbstractPacingType of the streams expression
    pub exp_pacing: TcKey,
    /// First element is the key to the AbstractPacingType of the spawn expression
    /// Second element is the key to the AbstractExpressionType of the spawn condition
    pub spawn: (TcKey, TcKey),
    /// The key to the AbstractExpressionType of the filter expression
    pub filter: TcKey,
    /// The key to the AbstractExpressionType of the close expression
    pub close: TcKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PacingErrorKind {
    FreqAnnotationNeeded(Span),
    NeverEval(Span),
    MalformedAC(Span, String),
    MixedEventPeriodic(AbstractPacingType, AbstractPacingType),
    IncompatibleExpressions(AbstractExpressionType, AbstractExpressionType),
    ParameterizationNeeded {
        who: Span,
        why: Span,
        spawn: Option<(AbstractPacingType, AbstractExpressionType)>,
        filter: Option<AbstractExpressionType>,
        close: Option<AbstractExpressionType>,
    },
    ParameterizationNotAllowed(Span),
    UnintuitivePacingWarning(Span, AbstractPacingType),
    Other(Span, String),
    OtherPacingError(Option<Span>, String, Vec<AbstractPacingType>),
    OtherExpressionError(Option<Span>, String, Vec<AbstractExpressionType>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PacingError {
    pub(crate) kind: PacingErrorKind,
    pub(crate) key1: Option<TcKey>,
    pub(crate) key2: Option<TcKey>,
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

/// The external definition of the stream pacing
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ConcreteStreamPacing {
    /// The pacing of the stream expression
    pub expression_pacing: ConcretePacingType,
    /// First element is the pacing of the spawn expression
    /// Second element is the spawn condition expression
    pub spawn: (ConcretePacingType, Expression),
    /// The filter expression
    pub filter: Expression,
    /// The close expression
    pub close: Expression,
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
                            Err(PacingErrorKind::MalformedAC(
                                ast_expr.span.clone(),
                                "Only 'True' is supported as literals in activation conditions.".into(),
                            )
                            .into())
                        }
                    }
                    _ => Err(PacingErrorKind::MalformedAC(
                        ast_expr.span.clone(),
                        "Only 'True' is supported as literals in activation conditions.".into(),
                    )
                    .into()),
                },
            },
            StreamAccess(sref, kind, args) => {
                if !args.is_empty() {
                    return Err(PacingErrorKind::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    )
                    .into());
                }
                match kind {
                    StreamAccessKind::Sync => {}
                    _ => {
                        return Err(PacingErrorKind::MalformedAC(
                            ast_expr.span.clone(),
                            "An activation condition can only contain literals and binary operators.".into(),
                        )
                        .into());
                    }
                }
                if sref.is_output() {
                    return Err(PacingErrorKind::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only refer to input streams".into(),
                    )
                    .into());
                }
                Ok(ActivationCondition::Stream(*sref))
            }
            ArithLog(op, v) => {
                if v.len() != 2 {
                    return Err(PacingErrorKind::MalformedAC(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    )
                    .into());
                }
                let ac_l = Self::parse(&v[0])?;
                let ac_r = Self::parse(&v[1])?;
                use crate::hir::expression::ArithLogOp;
                match op {
                    ArithLogOp::And | ArithLogOp::BitAnd => Ok(ac_l.and(ac_r)),
                    ArithLogOp::Or | ArithLogOp::BitOr => Ok(ac_l.or(ac_r)),
                    _ => Err(PacingErrorKind::MalformedAC(
                        ast_expr.span.clone(),
                        "Only '&' (and) or '|' (or) are allowed in activation conditions.".into(),
                    )
                    .into()),
                }
            }
            _ => Err(PacingErrorKind::MalformedAC(
                ast_expr.span.clone(),
                "An activation condition can only contain literals and binary operators.".into(),
            )
            .into()),
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
    pub(crate) fn emit(
        &self,
        handler: &Handler,
        pacing_spans: &HashMap<TcKey, Span>,
        exp_spans: &HashMap<TcKey, Span>,
        names: &HashMap<StreamReference, &str>,
    ) {
        use PacingErrorKind::*;
        match self.kind.clone() {
            FreqAnnotationNeeded(span) => {
                handler.error_with_span("In pacing type analysis:\nFrequency annotation needed.", span, Some("here"));
            }
            NeverEval(span) => {
                Diagnostic::error(handler, "In pacing type analysis:\nThe following stream is never evaluated.")
                    .add_span_with_label(span, Some("here"), true)
                    .add_note("Help: Consider annotating a pacing type explicitly.")
                    .emit();
            }
            MalformedAC(span, reason) => {
                handler.error_with_span(
                    &format!("In pacing type analysis:\nMalformed activation condition: {}", reason),
                    span,
                    Some("here"),
                );
            }
            MixedEventPeriodic(absty1, absty2) => {
                let span1 = self.key1.and_then(|k| pacing_spans.get(&k).cloned());
                let span2 = self.key2.and_then(|k| pacing_spans.get(&k).cloned());
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
            IncompatibleExpressions(e1, e2) => {
                let span1 = self.key1.and_then(|k| exp_spans.get(&k).cloned());
                let span2 = self.key2.and_then(|k| exp_spans.get(&k).cloned());
                Diagnostic::error(
                    handler,
                    format!(
                        "In pacing type analysis:\nIncompatible expressions: {} and {}",
                        e1.pretty_string(names),
                        e2.pretty_string(names)
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(
                    span1,
                    Some(format!("Found {} here", e1.pretty_string(names)).as_str()),
                    true,
                )
                .maybe_add_span_with_label(
                    span2,
                    Some(format!("and found {} here", e2.pretty_string(names)).as_str()),
                    false,
                )
                .emit();
            }
            UnintuitivePacingWarning(span, inferred) => Diagnostic::warning(
                handler,
                format!("In pacing type analysis:\nInferred complex pacing type: {}", inferred.to_string(names))
                    .as_str(),
            )
            .add_span_with_label(span, Some("here"), true)
            .add_note(
                format!(
                    "Help: Consider annotating the type explicitly for better readability using: @{}",
                    inferred.to_string(names)
                )
                .as_str(),
            )
            .emit(),
            Other(span, reason) => {
                handler.error_with_span(format!("In pacing type analysis:\n{}", reason).as_str(), span, Some("here"));
            }
            ParameterizationNotAllowed(span) => {
                Diagnostic::error(
                    handler,
                    "In pacing type analysis:\nSynchronous access to a parameterized stream is not allowed here.",
                )
                .add_span_with_label(span, Some("here"), true)
                .add_note("Help: Consider using a hold access")
                .emit();
            }
            ParameterizationNeeded { who, why, spawn, filter, close } => {
                let spawn_str = spawn.map_or("".into(), |(pacing, cond)| {
                    format!(
                        "\nspawn @{} with <...>{}",
                        pacing.to_string(names),
                        match cond {
                            AbstractExpressionType::Any => "".into(),
                            AbstractExpressionType::Expression(e) => format!(" if {}", e.pretty_string(names)),
                        }
                    )
                });
                let filter_str: String =
                    filter.map_or("".into(), |filter| format!("\nfilter {}", filter.pretty_string(names)));
                let close_str: String =
                    close.map_or("".into(), |close| format!("\nclose {}", close.pretty_string(names)));
                Diagnostic::error(handler, "In pacing type analysis:\nParameterization needed")
                    .add_span_with_label(who, Some("here"), true)
                    .add_span_with_label(why, Some("As of synchronous access occurring here"), false)
                    .add_note(&format!(
                        "Help: Consider adding the following template annotations:{}{}{}",
                        spawn_str, filter_str, close_str,
                    ))
                    .emit();
            }
            OtherPacingError(span, reason, causes) => {
                Diagnostic::error(
                    handler,
                    format!(
                        "In pacing type analysis:\n{} {}",
                        reason,
                        causes.iter().map(|ty| ty.to_string(names)).join(" and ")
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(span, Some("here"), true)
                .maybe_add_span_with_label(self.key1.and_then(|k| pacing_spans.get(&k).cloned()), Some("here"), true)
                .maybe_add_span_with_label(self.key2.and_then(|k| pacing_spans.get(&k).cloned()), Some("here"), true)
                .emit();
            }
            OtherExpressionError(span, reason, causes) => {
                Diagnostic::error(
                    handler,
                    format!(
                        "In pacing type analysis:\n{} {}",
                        reason,
                        causes.iter().map(|ty| ty.pretty_string(names)).join(" and ")
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(span, Some("here"), true)
                .maybe_add_span_with_label(self.key1.and_then(|k| exp_spans.get(&k).cloned()), Some("here"), true)
                .maybe_add_span_with_label(self.key2.and_then(|k| exp_spans.get(&k).cloned()), Some("here"), true)
                .emit();
            }
        }
    }
}

impl From<TcErr<AbstractPacingType>> for PacingError {
    fn from(error: TcErr<AbstractPacingType>) -> Self {
        match error {
            TcErr::KeyEquation(k1, k2, err) => PacingError { kind: err, key1: Some(k1), key2: Some(k2) },
            TcErr::Bound(k1, k2, err) => PacingError { kind: err, key1: Some(k1), key2: k2 },
            TcErr::ChildAccessOutOfBound(key, ty, _idx) => {
                let msg = "Child type out of bounds for type: ";
                PacingError {
                    kind: PacingErrorKind::OtherPacingError(None, msg.into(), vec![ty]),
                    key1: Some(key),
                    key2: None,
                }
            }
            TcErr::ArityMismatch { key, variant, inferred_arity, reported_arity } => {
                let msg = format!("Expected an arity of {} but got {} for type: ", inferred_arity, reported_arity);
                PacingError {
                    kind: PacingErrorKind::OtherPacingError(None, msg, vec![variant]),
                    key1: Some(key),
                    key2: None,
                }
            }
            TcErr::Construction(key, preliminary, kind) => PacingError { kind, key1: Some(key), key2: None },
            TcErr::ChildConstruction(key, idx, preliminary, kind) => {
                PacingError { kind, key1: Some(key), key2: preliminary.children[idx] }
            }
        }
    }
}

impl Into<PacingError> for PacingErrorKind {
    fn into(self) -> PacingError {
        PacingError { kind: self, key1: None, key2: None }
    }
}

impl From<TcErr<AbstractExpressionType>> for PacingError {
    fn from(error: TcErr<AbstractExpressionType>) -> Self {
        match error {
            TcErr::KeyEquation(k1, k2, err) => PacingError { kind: err, key1: Some(k1), key2: Some(k2) },
            TcErr::Bound(k1, k2, err) => PacingError { kind: err, key1: Some(k1), key2: k2 },
            TcErr::ChildAccessOutOfBound(key, ty, _idx) => {
                let msg = "Child type out of bounds for type: ";
                PacingError {
                    kind: PacingErrorKind::OtherExpressionError(None, msg.into(), vec![ty]),
                    key1: Some(key),
                    key2: None,
                }
            }
            TcErr::ArityMismatch { key, variant, inferred_arity, reported_arity } => {
                let msg = format!("Expected an arity of {} but got {} for type: ", inferred_arity, reported_arity);
                PacingError {
                    kind: PacingErrorKind::OtherExpressionError(None, msg, vec![variant]),
                    key1: Some(key),
                    key2: None,
                }
            }
            TcErr::Construction(key, preliminary, kind) => PacingError { kind, key1: Some(key), key2: None },
            TcErr::ChildConstruction(key, idx, preliminary, kind) => {
                PacingError { kind, key1: Some(key), key2: preliminary.children[idx] }
            }
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
            None => Err(PacingErrorKind::Other(
                Span::Unknown,
                format!("division of frequencies `{:?}`/`{:?}` failed", &lhs, &rhs),
            )
            .into()),
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

impl Variant for AbstractPacingType {
    type Err = PacingErrorKind;

    fn meet(lhs: Partial<Self>, rhs: Partial<Self>) -> Result<Partial<Self>, Self::Err> {
        use AbstractPacingType::*;
        assert_eq!(lhs.least_arity, 0, "Suspicious child");
        assert_eq!(rhs.least_arity, 0, "Suspicious child");
        let new_var = match (lhs.variant, rhs.variant) {
            (Any, x) | (x, Any) => Ok(x.clone()),
            (Event(ac), Periodic(f)) => Err(PacingErrorKind::MixedEventPeriodic(Event(ac.clone()), Periodic(*f))),
            (Periodic(f), Event(ac)) => Err(PacingErrorKind::MixedEventPeriodic(Periodic(*f), Event(ac.clone()))),
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
        }?;
        Ok(Partial { variant: new_var, least_arity: 0 })
    }

    fn fixed_arity(&self) -> bool {
        true
    }

    fn top() -> Self {
        AbstractPacingType::Any
    }
}

impl Constructable for AbstractPacingType {
    type Type = ConcretePacingType;

    fn construct(&self, children: &[Self::Type]) -> Result<Self::Type, Self::Err> {
        assert!(children.is_empty(), "Suspicious children");
        match self {
            AbstractPacingType::Any => Ok(ConcretePacingType::Constant),
            AbstractPacingType::Event(ac) => Ok(ConcretePacingType::Event(ac.clone())),
            AbstractPacingType::Periodic(freq) => match freq {
                Freq::Fixed(f) => Ok(ConcretePacingType::FixedPeriodic(f)),
                Freq::Any => Ok(ConcretePacingType::Periodic),
            },
        }
    }
}

impl AbstractPacingType {
    pub(crate) fn to_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            AbstractPacingType::Event(ac) => ac.to_string(names),
            AbstractPacingType::Periodic(freq) => freq.to_string(),
            AbstractPacingType::Any => "Any".to_string(),
        }
    }

    pub(crate) fn from_ac<M: HirMode + WithIrExpr + 'static>(
        ac: &AC,
        hir: &RTLolaHIR<M>,
    ) -> Result<(Self, Span), PacingErrorKind> {
        Ok(match ac {
            AC::Frequency { span, value } => (AbstractPacingType::Periodic(Freq::Fixed(*value)), span.clone()),
            AC::Expr(eid) => {
                let expr = hir.expression(*eid);
                (AbstractPacingType::Event(ActivationCondition::parse(expr)?), expr.span.clone())
            }
        })
    }
}

impl Variant for AbstractExpressionType {
    type Err = PacingErrorKind;

    fn meet(lhs: Partial<Self>, rhs: Partial<Self>) -> Result<Partial<Self>, Self::Err> {
        assert_eq!(lhs.least_arity, 0, "Suspicious child");
        assert_eq!(rhs.least_arity, 0, "Suspicious child");
        let new_var = match (lhs.variant, rhs.variant) {
            (Self::Any, x) | (x, Self::Any) => Ok(x.clone()),
            (Self::AnyClose, x) | (x, Self::AnyClose) => Ok(x.clone()),
            (Self::Expression(a), Self::Expression(b)) => {
                if a == b {
                    Ok(Self::Expression(a.clone()))
                } else {
                    Err(PacingErrorKind::IncompatibleExpressions(
                        Self::Expression(a.clone()),
                        Self::Expression(b.clone()),
                    ))
                }
            }
        }?;
        Ok(Partial { variant: new_var, least_arity: 0 })
    }

    fn fixed_arity(&self) -> bool {
        true
    }

    fn top() -> Self {
        AbstractExpressionType::Any
    }
}

impl Constructable for AbstractExpressionType {
    type Type = AbstractExpressionType;

    fn construct(&self, children: &[Self::Type]) -> Result<Self::Type, Self::Err> {
        assert!(children.is_empty(), "Suspicious children");

        match self {
            Self::Any => Ok(Expression {
                kind: ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true))),
                eid: ExprId(u32::max_value()),
                span: Span::Unknown,
            }),
            Self::AnyClose => Ok(Expression {
                kind: ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false))),
                eid: ExprId(u32::max_value()),
                span: Span::Unknown,
            }),
            Self::Expression(e) => Ok(e.clone()),
        }
        Ok(self.clone())
    }
}

impl std::fmt::Display for AbstractExpressionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "Any"),
            Self::AnyClose => write!(f, "AnyClose"),
            Self::Expression(e) => write!(f, "Exp({})", e),
        }
    }
}

impl AbstractExpressionType {
    fn pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            Self::Any => "Any".into(),
            Self::AnyClose => "AnyClose".into(),
            Self::Expression(e) => e.pretty_string(names),
        }
    }
}

impl ConcretePacingType {
    pub(crate) fn to_abstract_freq(&self) -> Result<AbstractPacingType, String> {
        match self {
            ConcretePacingType::FixedPeriodic(f) => Ok(AbstractPacingType::Periodic(Freq::Fixed(*f))),
            ConcretePacingType::Periodic => Ok(AbstractPacingType::Periodic(Freq::Any)),
            _ => Err("Supplied invalid concrete pacing type.".to_string()),
        }
    }

    pub(crate) fn from_ac<M: HirMode + WithIrExpr + 'static>(
        ac: &AC,
        hir: &RTLolaHIR<M>,
    ) -> Result<Self, PacingErrorKind> {
        Ok(match ac {
            AC::Frequency { span, value } => ConcretePacingType::FixedPeriodic(*value),
            AC::Expr(eid) => {
                let expr = hir.expression(*eid);
                ActivationCondition::parse(expr).map(|pac| ConcretePacingType::Event(pac))
            }
        })
    }
}
