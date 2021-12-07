use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use itertools::Itertools;
use num::{CheckedDiv, Integer};
use rtlola_reporting::{Diagnostic, Span};
use rusttyc::types::ContextSensitiveVariant;
use rusttyc::{Arity, Constructable, Partial, TcErr, TcKey, Variant};
use uom::lib::collections::HashMap;
use uom::lib::fmt::Formatter;
use uom::num_rational::Ratio;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;

use crate::hir::{
    AnnotatedPacingType, ArithLogOp, Constant, ExprId, Expression, ExpressionContext, ExpressionKind, FnExprKind, Hir,
    Inlined, Literal, StreamAccessKind, StreamReference, ValueEq, WidenExprKind,
};
use crate::modes::HirMode;
use crate::type_check::rtltc::{Resolvable, TypeError};
use crate::type_check::ConcretePacingType;

/// The activation condition describes when an event-based stream produces a new value.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Ord, Hash)]
pub enum ActivationCondition {
    /// When all of the activation conditions is true.
    Conjunction(Vec<Self>),
    /// When one of the activation conditions is true.
    Disjunction(Vec<Self>),
    /// Whenever the specified stream produces a new value.
    Stream(StreamReference),
    /// Whenever an event-based stream produces a new value.
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

#[derive(Debug, Clone)]
pub(crate) struct HashableExpression {
    context: Rc<ExpressionContext>,
    expression: Expression,
}

impl PartialEq for HashableExpression {
    fn eq(&self, other: &Self) -> bool {
        self.expression.value_eq(&other.expression, self.context.as_ref())
    }
}

impl Eq for HashableExpression {}

fn hash_expr_kind<H: Hasher>(kind: &ExpressionKind, state: &mut H) {
    match kind {
        ExpressionKind::LoadConstant(c) => {
            1.hash(state);
            c.hash(state);
        },
        ExpressionKind::ArithLog(op, args) => {
            2.hash(state);
            op.hash(state);
            args.iter().for_each(|arg| hash_expr_kind(&arg.kind, state));
        },
        ExpressionKind::StreamAccess(target, kind, _) => {
            // ignore parameters to fulfill:
            // key1 == key2 -> Hash(key1) == Hash(key2)
            3.hash(state);
            target.hash(state);
            kind.hash(state);
        },
        ExpressionKind::ParameterAccess(_, _) => {
            // ignore actual parameter <- See above
            4.hash(state);
        },
        ExpressionKind::Ite {
            condition,
            consequence,
            alternative,
        } => {
            5.hash(state);
            hash_expr_kind(&condition.kind, state);
            hash_expr_kind(&consequence.kind, state);
            hash_expr_kind(&alternative.kind, state);
        },
        ExpressionKind::Tuple(children) => {
            6.hash(state);
            children.iter().for_each(|child| hash_expr_kind(&child.kind, state))
        },
        ExpressionKind::TupleAccess(target, idx) => {
            7.hash(state);
            hash_expr_kind(&target.kind, state);
            idx.hash(state);
        },
        ExpressionKind::Function(func_def) => {
            8.hash(state);
            let FnExprKind { name, args, type_param } = &func_def;
            name.hash(state);
            args.iter().for_each(|arg| hash_expr_kind(&arg.kind, state));
            type_param.hash(state);
        },
        ExpressionKind::Widen(widen_kind) => {
            9.hash(state);
            let WidenExprKind { expr, ty } = &widen_kind;
            hash_expr_kind(&expr.kind, state);
            ty.hash(state);
        },
        ExpressionKind::Default { expr, default } => {
            10.hash(state);
            hash_expr_kind(&expr.kind, state);
            hash_expr_kind(&default.kind, state);
        },
    }
}

impl Hash for HashableExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_expr_kind(&self.expression.kind, state);
    }
}

/// The internal representation of an expression type
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AbstractExpressionType {
    /// Any is concretized into True
    Any,
    /// AnyClose is concretized into False
    AnyClose,
    /// A conjunction of expressions.
    Conjunction(HashSet<HashableExpression>),
    /// A disjunction of expressions
    Disjunction(HashSet<HashableExpression>),
    /// A single expression that is neither a conjunction nor a disjunction.
    Mixed(HashableExpression),
}

/// The internal representation of the overall Stream pacing
/// Types are given by keys in the respective rustic instances
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamTypeKeys {
    /// Key to the AbstractPacingType of the streams expression
    pub(crate) exp_pacing: TcKey,
    /// First element is the key to the AbstractPacingType of the spawn expression
    /// Second element is the key to the AbstractExpressionType of the spawn condition
    pub(crate) spawn: (TcKey, TcKey),
    /// The key to the AbstractExpressionType of the filter expression
    pub(crate) filter: TcKey,
    /// The key to the AbstractExpressionType of the close expression
    pub(crate) close: TcKey,
}

/// Reference for stream  template during pacing type inference, used in error reporting.
#[derive(Debug)]
pub(crate) struct InferredTemplates {
    pub(crate) spawn_pacing: Option<ConcretePacingType>,
    pub(crate) spawn_cond: Option<Expression>,
    pub(crate) filter: Option<Expression>,
    pub(crate) close: Option<Expression>,
}

/// The [PacingErrorKind] helps to distinguish errors
/// during reporting.
#[derive(Debug)]
pub(crate) enum PacingErrorKind {
    FreqAnnotationNeeded(Span),
    NeverEval(Span),
    MalformedAc(Span, String),
    MixedEventPeriodic(AbstractPacingType, AbstractPacingType),
    IncompatibleExpressions(AbstractExpressionType, AbstractExpressionType),
    ParameterizationNeeded {
        who: Span,
        why: Span,
        inferred: Box<InferredTemplates>,
    },
    /// Bound, Inferred
    PacingTypeMismatch(ConcretePacingType, ConcretePacingType),
    ParameterizationNotAllowed(Span),
    UnintuitivePacingWarning(Span, ConcretePacingType),
    Other(Span, String, Vec<Box<dyn PrintableVariant>>),
    SpawnPeriodicMismatch(Span, Span, (ConcretePacingType, Expression)),
    InvalidSyncAccessParameter {
        target_span: Span,
        target_spawn_expr: Expression,
        own_spawn_expr: Expression,
        arg: Expression,
    },
    NonParamInSyncAccess(Span),
    ParameterAmountMismatch {
        target_span: Span,
        exp_span: Span,
        given_num: usize,
        expected_num: usize,
    },
}

impl std::ops::BitAnd for ActivationCondition {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ActivationCondition::Conjunction(mut left), ActivationCondition::Conjunction(mut right)) => {
                left.append(&mut right);
                left.sort();
                left.dedup();
                ActivationCondition::Conjunction(left)
            },
            (ActivationCondition::True, other) | (other, ActivationCondition::True) => other,
            (ActivationCondition::Conjunction(mut other_con), other_ac)
            | (other_ac, ActivationCondition::Conjunction(mut other_con)) => {
                other_con.push(other_ac);
                other_con.sort();
                other_con.dedup();
                ActivationCondition::Conjunction(other_con)
            },
            (a, b) => {
                let mut childs = vec![a, b];
                childs.sort();
                childs.dedup();
                ActivationCondition::Conjunction(childs)
            },
        }
        .flatten()
    }
}

impl std::ops::BitOr for ActivationCondition {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ActivationCondition::Disjunction(mut left), ActivationCondition::Disjunction(mut right)) => {
                left.append(&mut right);
                left.sort();
                left.dedup();
                ActivationCondition::Disjunction(left)
            },
            (ActivationCondition::True, _) | (_, ActivationCondition::True) => ActivationCondition::True,
            (ActivationCondition::Disjunction(mut other_dis), other_ac)
            | (other_ac, ActivationCondition::Disjunction(mut other_dis)) => {
                other_dis.push(other_ac);
                other_dis.sort();
                other_dis.dedup();
                ActivationCondition::Disjunction(other_dis)
            },
            (a, b) => {
                let mut childs = vec![a, b];
                childs.sort();
                childs.dedup();
                ActivationCondition::Disjunction(childs)
            },
        }
        .flatten()
    }
}
impl ActivationCondition {
    /// Flattens the [ActivationCondition] if the Conjunction/Disjunction contains only a single element. Does nothing otherwise.
    /// # Example
    ///  Conjunction(a).flatten() => a
    ///  Disjunction(a,b).flatten() => Disjunction(a,b)
    pub fn flatten(self) -> Self {
        match self {
            ActivationCondition::Conjunction(mut v) | ActivationCondition::Disjunction(mut v) if v.len() == 1 => {
                v.remove(0)
            },
            _ => self,
        }
    }

    fn parse(ast_expr: &Expression) -> Result<Self, PacingErrorKind> {
        use ExpressionKind::*;
        match &ast_expr.kind {
            LoadConstant(c) => {
                match c {
                    Constant::Basic(lit) | Constant::Inlined(Inlined { lit, .. }) => {
                        match lit {
                            Literal::Bool(b) => {
                                if *b {
                                    Ok(ActivationCondition::True)
                                } else {
                                    Err(PacingErrorKind::MalformedAc(
                                        ast_expr.span.clone(),
                                        "Only 'True' is supported as literals in activation conditions.".into(),
                                    ))
                                }
                            },
                            _ => {
                                Err(PacingErrorKind::MalformedAc(
                                    ast_expr.span.clone(),
                                    "Only 'True' is supported as literals in activation conditions.".into(),
                                ))
                            },
                        }
                    },
                }
            },
            StreamAccess(sref, kind, args) => {
                if !args.is_empty() {
                    return Err(PacingErrorKind::MalformedAc(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    ));
                }
                match kind {
                    StreamAccessKind::Sync => {},
                    _ => {
                        return Err(PacingErrorKind::MalformedAc(
                            ast_expr.span.clone(),
                            "An activation condition can only contain literals and binary operators.".into(),
                        ));
                    },
                }
                if sref.is_output() {
                    return Err(PacingErrorKind::MalformedAc(
                        ast_expr.span.clone(),
                        "An activation condition can only refer to input streams".into(),
                    ));
                }
                Ok(ActivationCondition::Stream(*sref))
            },
            ArithLog(op, v) => {
                if v.len() != 2 {
                    return Err(PacingErrorKind::MalformedAc(
                        ast_expr.span.clone(),
                        "An activation condition can only contain literals and binary operators.".into(),
                    ));
                }
                let ac_l = Self::parse(&v[0])?;
                let ac_r = Self::parse(&v[1])?;
                match op {
                    ArithLogOp::And | ArithLogOp::BitAnd => Ok(ac_l & ac_r),
                    ArithLogOp::Or | ArithLogOp::BitOr => Ok(ac_l | ac_r),
                    _ => {
                        Err(PacingErrorKind::MalformedAc(
                            ast_expr.span.clone(),
                            "Only '&' (and) or '|' (or) are allowed in activation conditions.".into(),
                        ))
                    },
                }
            },
            _ => {
                Err(PacingErrorKind::MalformedAc(
                    ast_expr.span.clone(),
                    "An activation condition can only contain literals and binary operators.".into(),
                ))
            },
        }
    }

    /// Print function for [ActivationCondition]. Used for error reporting prints.
    pub fn to_string(&self, stream_names: &HashMap<StreamReference, &str>) -> String {
        use ActivationCondition::*;
        match self {
            True => "⊤".into(),
            Stream(sr) => stream_names[sr].into(),
            Conjunction(childs) => {
                let child_string: String = childs.iter().map(|ac| ac.to_string(stream_names)).join(" ∧ ");
                format!("({})", child_string)
            },
            Disjunction(childs) => {
                let child_string: String = childs.iter().map(|ac| ac.to_string(stream_names)).join(" ∨ ");
                format!("({})", child_string)
            },
        }
    }
}

impl Resolvable for PacingErrorKind {
    fn into_diagnostic(
        self,
        spans: &[&HashMap<TcKey, Span>],
        names: &HashMap<StreamReference, &str>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    ) -> Diagnostic {
        let pacing_spans = spans[0];
        let exp_spans = spans[1];
        use PacingErrorKind::*;
        match self {
            FreqAnnotationNeeded(span) => {
                Diagnostic::error("In pacing type analysis:\nFrequency annotation needed.").add_span_with_label(
                    span,
                    Some("here"),
                    true,
                )
            },
            NeverEval(span) => {
                Diagnostic::error("In pacing type analysis:\nThe following stream or expression is never evaluated.")
                    .add_span_with_label(span, Some("here"), true)
                    .add_note("Help: Consider annotating a pacing type explicitly.")
            },
            MalformedAc(span, reason) => {
                Diagnostic::error(&format!(
                    "In pacing type analysis:\nMalformed activation condition: {}",
                    reason
                ))
                .add_span_with_label(span, Some("here"), true)
            },
            MixedEventPeriodic(absty1, absty2) => {
                let span1 = key1.and_then(|k| pacing_spans.get(&k).cloned());
                let span2 = key2.and_then(|k| pacing_spans.get(&k).cloned());
                let ty1 = absty1.to_pretty_string(names);
                let ty2 = absty2.to_pretty_string(names);
                Diagnostic::error(
                    format!(
                        "In pacing type analysis:\nMixed an event and a periodic type: {} and {}",
                        ty1, ty2
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(span1, Some(format!("Found {} here", ty1).as_str()), true)
                .maybe_add_span_with_label(
                    span2,
                    Some(format!("and found {} here", ty2).as_str()),
                    false,
                )
            },
            IncompatibleExpressions(e1, e2) => {
                let span1 = key1.and_then(|k| exp_spans.get(&k).cloned());
                let span2 = key2.and_then(|k| exp_spans.get(&k).cloned());
                Diagnostic::error(
                    format!(
                        "In pacing type analysis:\nIncompatible expressions: {} and {}",
                        e1.to_pretty_string(names),
                        e2.to_pretty_string(names)
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(
                    span1,
                    Some(format!("Found {} here", e1.to_pretty_string(names)).as_str()),
                    true,
                )
                .maybe_add_span_with_label(
                    span2,
                    Some(format!("and found {} here", e2.to_pretty_string(names)).as_str()),
                    false,
                )
            },
            UnintuitivePacingWarning(span, inferred) => {
                Diagnostic::warning(
                    format!(
                        "In pacing type analysis:\nInferred complex pacing type: {}",
                        inferred.to_pretty_string(names)
                    )
                    .as_str(),
                )
                .add_span_with_label(span, Some("here"), true)
                .add_note(
                    format!(
                        "Help: Consider annotating the type explicitly for better readability using: @{}",
                        inferred.to_pretty_string(names)
                    )
                    .as_str(),
                )
            },
            Other(span, reason, causes) => {
                Diagnostic::error(
                    format!(
                        "In pacing type analysis:\n{} {}",
                        reason,
                        causes.iter().map(|ty| ty.to_pretty_string(names)).join(" and ")
                    )
                    .as_str(),
                )
                .add_span_with_label(span, Some("here"), true)
                .maybe_add_span_with_label(key1.and_then(|k| pacing_spans.get(&k).cloned()), Some("here"), true)
                .maybe_add_span_with_label(
                    key2.and_then(|k| pacing_spans.get(&k).cloned()),
                    Some("here"),
                    true,
                )
            },
            ParameterizationNotAllowed(span) => {
                Diagnostic::error(
                    "In pacing type analysis:\nSynchronous access to a parameterized stream is not allowed here.",
                )
                .add_span_with_label(span, Some("here"), true)
                .add_note("Help: Consider using a hold access")
            },
            ParameterizationNeeded { who, why, inferred } => {
                let InferredTemplates {
                    spawn_pacing,
                    spawn_cond,
                    filter,
                    close,
                } = *inferred;
                let spawn_str = match (spawn_pacing, spawn_cond) {
                    (Some(pacing), Some(cond)) => {
                        format!(
                            "\nspawn @{} with <...> if {}",
                            pacing.to_pretty_string(names),
                            cond.pretty_string(names)
                        )
                    },
                    (Some(pacing), None) => format!("\nspawn @{} with <...>", pacing.to_pretty_string(names)),
                    (None, Some(cond)) => format!("\nspawn <...> if {}", cond.pretty_string(names)),
                    (None, None) => "".to_string(),
                };
                let filter_str: String =
                    filter.map_or("".into(), |filter| format!("\nfilter {}", filter.pretty_string(names)));
                let close_str: String =
                    close.map_or("".into(), |close| format!("\nclose {}", close.pretty_string(names)));
                Diagnostic::error("In pacing type analysis:\nParameterization needed")
                    .add_span_with_label(who, Some("here"), true)
                    .add_span_with_label(why, Some("As of synchronous access occurring here"), false)
                    .add_note(&format!(
                        "Help: Consider adding the following template annotations:{}{}{}",
                        spawn_str, filter_str, close_str,
                    ))
            },
            PacingTypeMismatch(bound, inferred) => {
                let bound_str = bound.to_pretty_string(names);
                let inferred_str = inferred.to_pretty_string(names);
                let bound_span = key1.map(|k| pacing_spans[&k].clone());
                let inferred_span = key2.and_then(|k| pacing_spans.get(&k).cloned());
                Diagnostic::error(
                    format!(
                        "In pacing type analysis:\nInferred pacing type: {} but expected: {}",
                        &inferred_str, &bound_str
                    )
                    .as_str(),
                )
                .maybe_add_span_with_label(bound_span, Some(format!("Expected {} here", bound_str).as_str()), true)
                .maybe_add_span_with_label(
                    inferred_span,
                    Some(format!("Inferred {} here", inferred_str).as_str()),
                    true,
                )
            },
            SpawnPeriodicMismatch(access_span, target_span, (access_pacing, access_condition)) => Diagnostic::error(
                "In pacing type analysis:\nPeriodic stream out of sync with accessed stream due to a spawn annotation.",
            )
            .add_span_with_label(
                access_span,
                Some(
                    format!(
                        "Found accessing stream here with: spawn @{} <...> if {}",
                        access_pacing.to_pretty_string(names),
                        access_condition.pretty_string(names)
                    )
                    .as_str(),
                ),
                true,
            )
            .add_span_with_label(target_span, Some("Found target stream here"), false),
            InvalidSyncAccessParameter {
                target_span,
                target_spawn_expr,
                own_spawn_expr,
                arg,
            } => {
                let target_expr = target_spawn_expr.pretty_string(names);
                let own_expr = own_spawn_expr.pretty_string(names);
                let supplied = arg.pretty_string(names);

                Diagnostic::error(
                    "In pacing type analysis:\nInvalid argument for synchronized access:"
                )
                .add_span_with_label(target_span, Some(&format!("Target expected the argument to be equal to the spawn expression: ({})", target_expr)), false)
                .add_span_with_label(arg.span, Some(&format!("Supplied arguments ({}) resolved to the spawn expressions: ({})",
                                                             supplied,
                    own_expr
                )), true)
                    .add_note("Note: Each parameter of the accessed stream requires a counterpart which is a parameter of the accessing stream.")
            },
            NonParamInSyncAccess(span) => {
                Diagnostic::error(
                                  "In pacing type analysis:\nOnly parameters are allowed as arguments when synchronously accessing a stream:"
                )
                    .add_span_with_label(span, Some("Found an expression that is not a parameter here"), true)
            },
            ParameterAmountMismatch { target_span, exp_span, given_num, expected_num} => {
                Diagnostic::error(
                                  "In pacing type analysis:\nMismatch between number of given arguments and expected spawn arguments:"
                )
                    .add_span_with_label(exp_span, Some(&format!("Got {} arguments here.", given_num)), true)
                    .add_span_with_label(target_span, Some(&format!("Expected {} arguments here.", expected_num)), false)
            }
        }
    }
}

pub(crate) trait PrintableVariant: Debug {
    fn to_pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String;
}

impl<V: 'static + ContextSensitiveVariant<Err = PacingErrorKind> + PrintableVariant> From<TcErr<V>>
    for TypeError<PacingErrorKind>
{
    fn from(err: TcErr<V>) -> TypeError<PacingErrorKind> {
        match err {
            TcErr::KeyEquation(k1, k2, err) => {
                TypeError {
                    kind: err,
                    key1: Some(k1),
                    key2: Some(k2),
                }
            },
            TcErr::Bound(k1, k2, err) => {
                TypeError {
                    kind: err,
                    key1: Some(k1),
                    key2: k2,
                }
            },
            TcErr::ChildAccessOutOfBound(key, ty, _idx) => {
                let msg = "Child type out of bounds for type: ";
                TypeError {
                    kind: PacingErrorKind::Other(Span::Unknown, msg.into(), vec![Box::new(ty)]),
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::ArityMismatch {
                key,
                variant,
                inferred_arity,
                reported_arity,
            } => {
                let msg = format!(
                    "Expected an arity of {} but got {} for type: ",
                    inferred_arity, reported_arity
                );
                TypeError {
                    kind: PacingErrorKind::Other(Span::Unknown, msg, vec![Box::new(variant)]),
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::Construction(key, _preliminary, kind) => {
                TypeError {
                    kind,
                    key1: Some(key),
                    key2: None,
                }
            },
            TcErr::ChildConstruction(key, idx, preliminary, kind) => {
                TypeError {
                    kind,
                    key1: Some(key),
                    key2: preliminary.children[idx],
                }
            },
            TcErr::CyclicGraph => {
                panic!("Cyclic pacing type constraint system");
            },
        }
    }
}

// Abstract Type Definition

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Freq::Any => write!(f, "Periodic"),
            Freq::Fixed(freq) => {
                write!(
                    f,
                    "{}",
                    (*freq).into_format_args(hertz, uom::fmt::DisplayStyle::Abbreviation)
                )
            },
        }
    }
}

impl Freq {
    /// Checks if there exists an k ∈ ℕ s.t. self =  k * other.
    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, PacingErrorKind> {
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
            None => {
                Err(PacingErrorKind::Other(
                    Span::Unknown,
                    format!("division of frequencies `{:?}`/`{:?}` failed", &lhs, &rhs),
                    vec![],
                ))
            },
        }
    }

    fn conjunction(&self, other: &Freq) -> Freq {
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

    fn top() -> Self {
        AbstractPacingType::Any
    }

    fn meet(lhs: Partial<Self>, rhs: Partial<Self>) -> Result<Partial<Self>, Self::Err> {
        use AbstractPacingType::*;
        assert_eq!(lhs.least_arity, 0, "suspicious child");
        assert_eq!(rhs.least_arity, 0, "suspicious child");
        // Todo: Avoid clone
        let new_var = match (lhs.variant.clone(), rhs.variant.clone()) {
            (Any, x) | (x, Any) => Ok(x),
            (Periodic(_), Event(_)) | (Event(_), Periodic(_)) => {
                Err(PacingErrorKind::MixedEventPeriodic(lhs.variant, rhs.variant))
            },
            (Event(ac1), Event(ac2)) => Ok(Event(ac1 & ac2)),
            (Periodic(f1), Periodic(f2)) => {
                if let Freq::Any = f1 {
                    Ok(Periodic(f2))
                } else if let Freq::Any = f2 {
                    Ok(Periodic(f1))
                } else {
                    Ok(Periodic(f1.conjunction(&f2)))
                }
            },
        }?;
        Ok(Partial {
            variant: new_var,
            least_arity: 0,
        })
    }

    fn arity(&self) -> Arity {
        Arity::Fixed(0)
    }
}

impl Constructable for AbstractPacingType {
    type Type = ConcretePacingType;

    fn construct(&self, children: &[Self::Type]) -> Result<Self::Type, Self::Err> {
        assert!(children.is_empty(), "Suspicious children");
        match self {
            AbstractPacingType::Any => Ok(ConcretePacingType::Constant),
            AbstractPacingType::Event(ac) => Ok(ConcretePacingType::Event(ac.clone())),
            AbstractPacingType::Periodic(freq) => {
                match freq {
                    Freq::Fixed(f) => Ok(ConcretePacingType::FixedPeriodic(*f)),
                    Freq::Any => Ok(ConcretePacingType::Periodic),
                }
            },
        }
    }
}

impl PrintableVariant for AbstractPacingType {
    fn to_pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            AbstractPacingType::Event(ac) => ac.to_string(names),
            AbstractPacingType::Periodic(freq) => freq.to_string(),
            AbstractPacingType::Any => "Any".to_string(),
        }
    }
}

impl PrintableVariant for AbstractExpressionType {
    fn to_pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            Self::Any => "Any".into(),
            Self::AnyClose => "AnyClose".into(),
            Self::Mixed(e) => format!("Mixed({})", e.expression),
            Self::Conjunction(conjs) => {
                format!(
                    "Conjunction({})",
                    conjs.iter().map(|he| he.expression.pretty_string(names)).join(", ")
                )
            },
            Self::Disjunction(disj) => {
                format!(
                    "Disjunction({})",
                    disj.iter().map(|he| he.expression.pretty_string(names)).join(", ")
                )
            },
        }
    }
}

impl AbstractPacingType {
    /// Transforms a given [Ac] (annotated in the [Hir]) into an abstract pacing type.
    pub(crate) fn from_pt<M: HirMode>(pt: &AnnotatedPacingType, hir: &Hir<M>) -> Result<(Self, Span), PacingErrorKind> {
        Ok(match pt {
            AnnotatedPacingType::Frequency { span, value } => {
                (AbstractPacingType::Periodic(Freq::Fixed(*value)), span.clone())
            },
            AnnotatedPacingType::Expr(eid) => {
                let expr = hir.expression(*eid);
                (
                    AbstractPacingType::Event(ActivationCondition::parse(expr)?),
                    expr.span.clone(),
                )
            },
        })
    }
}

impl Variant for AbstractExpressionType {
    type Err = PacingErrorKind;

    fn top() -> Self {
        AbstractExpressionType::Any
    }

    fn meet(lhs: Partial<Self>, rhs: Partial<Self>) -> Result<Partial<Self>, Self::Err> {
        assert_eq!(lhs.least_arity, 0, "suspicious child");
        assert_eq!(rhs.least_arity, 0, "suspicious child");
        let new_var = match (lhs.variant, rhs.variant) {
            (Self::Any, x) | (x, Self::Any) => Ok(x),
            (Self::AnyClose, x) | (x, Self::AnyClose) => Ok(x),
            _ => todo!(),
            /* (Self::Expression(a), Self::Expression(b)) => {
             *     if a.value_eq(&b, ctx) {
             *         Ok(Self::Expression(a))
             *     } else {
             *         Err(PacingErrorKind::IncompatibleExpressions(
             *             Self::Expression(a),
             *             Self::Expression(b),
             *         ))
             *     }
             * }, */
        }?;
        Ok(Partial {
            variant: new_var,
            least_arity: 0,
        })
    }

    fn arity(&self) -> Arity {
        Arity::Fixed(0)
    }
}

impl Constructable for AbstractExpressionType {
    type Type = Expression;

    fn construct(&self, children: &[Self::Type]) -> Result<Self::Type, Self::Err> {
        assert!(children.is_empty(), "suspicious children");

        match self {
            Self::Any => {
                Ok(Expression {
                    kind: ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(true))),
                    eid: ExprId(u32::MAX),
                    span: Span::Unknown,
                })
            },
            Self::AnyClose => {
                Ok(Expression {
                    kind: ExpressionKind::LoadConstant(Constant::Basic(Literal::Bool(false))),
                    eid: ExprId(u32::MAX),
                    span: Span::Unknown,
                })
            },
            _ => todo!(),
            // Self::Expression(e) => Ok(e.clone()),
        }
    }
}

impl std::fmt::Display for AbstractExpressionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "Any"),
            Self::AnyClose => write!(f, "AnyClose"),
            Self::Mixed(e) => write!(f, "Mixed({})", e.expression),
            Self::Conjunction(conjs) => {
                write!(
                    f,
                    "Conjunction({})",
                    conjs.iter().map(|he| format!("{}", he.expression)).join(", ")
                )
            },
            Self::Disjunction(disj) => {
                write!(
                    f,
                    "Disjunction({})",
                    disj.iter().map(|he| format!("{}", he.expression)).join(", ")
                )
            },
        }
    }
}

impl AbstractExpressionType {
    /// Joins self with other
    fn join<F>(self, other: Self, mixed_constructor: F) -> AbstractExpressionType
    where
        F: Fn(HashSet<HashableExpression>) -> AbstractExpressionType,
    {
        use AbstractExpressionType::*;
        match (self, other) {
            (Any, _)
            | (_, Any)
            | (AnyClose, _)
            | (_, AnyClose)
            | (Conjunction(_), Disjunction(_))
            | (Disjunction(_), Conjunction(_)) => panic!("Can only join Conjunctions, Disjunctions or Single"),
            (Mixed(a), Mixed(b)) => mixed_constructor(vec![a, b].into_iter().collect()),
            (Mixed(this), Conjunction(mut other)) | (Conjunction(mut other), Mixed(this)) => {
                other.insert(this);
                Conjunction(other)
            },
            (Mixed(this), Disjunction(mut other)) | (Disjunction(mut other), Mixed(this)) => {
                other.insert(this);
                Disjunction(other)
            },
            (Conjunction(mut this), Conjunction(other)) => {
                this.extend(other);
                Conjunction(this)
            },
            (Disjunction(mut this), Disjunction(other)) => {
                this.extend(other);
                Disjunction(this)
            },
        }
    }

    fn contains_and_or(exp: &Expression) -> bool {
        match &exp.kind {
            ExpressionKind::ArithLog(op, args) => {
                match op {
                    ArithLogOp::Not => Self::contains_and_or(&args[0]),
                    ArithLogOp::And | ArithLogOp::Or => true,
                    ArithLogOp::Sub
                    | ArithLogOp::Mul
                    | ArithLogOp::Div
                    | ArithLogOp::Rem
                    | ArithLogOp::Pow
                    | ArithLogOp::Add
                    | ArithLogOp::Neg
                    | ArithLogOp::BitXor
                    | ArithLogOp::BitAnd
                    | ArithLogOp::BitOr
                    | ArithLogOp::BitNot
                    | ArithLogOp::Shl
                    | ArithLogOp::Shr
                    | ArithLogOp::Eq
                    | ArithLogOp::Lt
                    | ArithLogOp::Le
                    | ArithLogOp::Ne
                    | ArithLogOp::Ge
                    | ArithLogOp::Gt => false,
                }
            },
            ExpressionKind::LoadConstant(_)
            | ExpressionKind::StreamAccess(_, _, _)
            | ExpressionKind::ParameterAccess(_, _)
            | ExpressionKind::Ite { .. }
            | ExpressionKind::Tuple(_)
            | ExpressionKind::TupleAccess(_, _)
            | ExpressionKind::Function(_)
            | ExpressionKind::Widen(_)
            | ExpressionKind::Default { .. } => false,
        }
    }

    /// Tries to parse the expression tree from a /\ (b /\ c) to conjunction(a,b,c)
    /// target determines whether a conjunction or disjunction is considered
    /// If conjunctions and disjunctions are mixed, Err is returned
    /// None -> Not determined yet
    /// Some(true) -> Conjunction
    /// Some(false) -> Disjunction
    fn parse_pure(exp: &Expression, target: Option<bool>, context: Rc<ExpressionContext>) -> Result<Self, ()> {
        match &exp.kind {
            ExpressionKind::ArithLog(op, args) => {
                match (op, target) {
                    (ArithLogOp::And, None) | (ArithLogOp::And, Some(true)) => {
                        let left = Self::parse_pure(&args[0], Some(true), context.clone())?;
                        let right = Self::parse_pure(&args[1], Some(true), context)?;
                        Ok(left.join(right, AbstractExpressionType::Conjunction))
                    },
                    (ArithLogOp::Or, None) | (ArithLogOp::Or, Some(false)) => {
                        let left = Self::parse_pure(&args[0], Some(false), context.clone())?;
                        let right = Self::parse_pure(&args[1], Some(false), context)?;
                        Ok(left.join(right, AbstractExpressionType::Disjunction))
                    },
                    (ArithLogOp::And, Some(false)) | (ArithLogOp::Or, Some(true)) => Err(()),
                    (ArithLogOp::Not, _) => {
                        if Self::contains_and_or(exp) {
                            Err(())
                        } else {
                            Ok(Self::Mixed(HashableExpression {
                                context,
                                expression: exp.clone(),
                            }))
                        }
                    },
                    (ArithLogOp::Neg, _)
                    | (ArithLogOp::Add, _)
                    | (ArithLogOp::Sub, _)
                    | (ArithLogOp::Mul, _)
                    | (ArithLogOp::Div, _)
                    | (ArithLogOp::Rem, _)
                    | (ArithLogOp::Pow, _)
                    | (ArithLogOp::BitXor, _)
                    | (ArithLogOp::BitAnd, _)
                    | (ArithLogOp::BitOr, _)
                    | (ArithLogOp::BitNot, _)
                    | (ArithLogOp::Shl, _)
                    | (ArithLogOp::Shr, _)
                    | (ArithLogOp::Eq, _)
                    | (ArithLogOp::Lt, _)
                    | (ArithLogOp::Le, _)
                    | (ArithLogOp::Ne, _)
                    | (ArithLogOp::Ge, _)
                    | (ArithLogOp::Gt, _) => {
                        Ok(Self::Mixed(HashableExpression {
                            context,
                            expression: exp.clone(),
                        }))
                    },
                }
            },
            ExpressionKind::LoadConstant(_)
            | ExpressionKind::Default { .. }
            | ExpressionKind::Widen(_)
            | ExpressionKind::Function(_)
            | ExpressionKind::TupleAccess(_, _)
            | ExpressionKind::Tuple(_)
            | ExpressionKind::Ite { .. }
            | ExpressionKind::StreamAccess(_, _, _)
            | ExpressionKind::ParameterAccess(_, _) => {
                Ok(Self::Mixed(HashableExpression {
                    context,
                    expression: exp.clone(),
                }))
            },
        }
    }

    pub(crate) fn from_expression(exp: &Expression, context: Rc<ExpressionContext>) -> Self {
        Self::parse_pure(exp, None, context.clone()).unwrap_or_else(|_| {
            AbstractExpressionType::Mixed(HashableExpression {
                context,
                expression: exp.clone(),
            })
        })
    }
}

impl ConcretePacingType {
    /// Pretty print function for [ConcretePacingType].
    pub fn to_pretty_string(&self, names: &HashMap<StreamReference, &str>) -> String {
        match self {
            ConcretePacingType::Event(ac) => ac.to_string(names),
            ConcretePacingType::FixedPeriodic(freq) => {
                (*freq)
                    .into_format_args(hertz, uom::fmt::DisplayStyle::Abbreviation)
                    .to_string()
            },
            ConcretePacingType::Periodic => "Periodic".to_string(),
            ConcretePacingType::Constant => "Constant".to_string(),
        }
    }

    /// Tries to convert a concrete pacing into a frequency.
    pub(crate) fn to_abstract_freq(&self) -> Result<AbstractPacingType, String> {
        match self {
            ConcretePacingType::FixedPeriodic(f) => Ok(AbstractPacingType::Periodic(Freq::Fixed(*f))),
            ConcretePacingType::Periodic => Ok(AbstractPacingType::Periodic(Freq::Any)),
            _ => Err("Supplied invalid concrete pacing type.".to_string()),
        }
    }

    /// Transforms a given [Ac] (annotated in the [Hir]) into a pacing type.
    pub(crate) fn from_pt<M: HirMode>(pt: &AnnotatedPacingType, hir: &Hir<M>) -> Result<Self, PacingErrorKind> {
        match pt {
            AnnotatedPacingType::Frequency { span: _, value } => Ok(ConcretePacingType::FixedPeriodic(*value)),
            AnnotatedPacingType::Expr(eid) => {
                let expr = hir.expression(*eid);
                ActivationCondition::parse(expr).map(ConcretePacingType::Event)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash};

    use rtlola_parser::{ParserConfig, RtLolaAst};

    use crate::hir::{ExpressionContext, ValueEq};
    use crate::type_check::pacing_types::{HashableExpression, AbstractExpressionType};
    use crate::{BaseMode, RtLolaHir};
    use std::rc::Rc;

    struct TestEnv {
        hir: RtLolaHir<BaseMode>,
        ctx: Rc<ExpressionContext>,
    }

    impl TestEnv {
        fn from_spec(spec: &str) -> Self {
            let ast: RtLolaAst = match rtlola_parser::parse(ParserConfig::for_string(spec.to_string())) {
                Ok(s) => s,
                Err(e) => panic!("Spec {} cannot be parsed: {:?}", spec, e),
            };
            let hir = crate::from_ast(ast).unwrap();

            let ctx = Rc::new(ExpressionContext::new(&hir));

            TestEnv { hir, ctx }
        }
    }

    #[test]
    fn test_expression_hash_eq() {
        let env = TestEnv::from_spec(
            "\
            input i: Int32\n\
            output a(p: Int32) spawn with i := i + p\n\
            output b(q: Int32) spawn with i := i + q",
        );
        let a_exp = env.hir.expression(env.hir.outputs[0].expr_id);
        let b_exp = env.hir.expression(env.hir.outputs[1].expr_id);
        assert!(a_exp.value_neq_ignore_parameters(b_exp));
        assert!(a_exp.value_eq(b_exp, env.ctx.as_ref()));
        let a_hash_expr = HashableExpression {
            context: env.ctx.clone(),
            expression: a_exp.clone(),
        };

        let b_hash_expr = HashableExpression {
            context: env.ctx,
            expression: b_exp.clone(),
        };

        let mut hasher_a = RandomState::new().build_hasher();
        let mut hasher_b = RandomState::new().build_hasher();
        assert_eq!(a_hash_expr, b_hash_expr);
        assert_eq!(a_hash_expr.hash(&mut hasher_a), b_hash_expr.hash(&mut hasher_b));
    }

    #[test]
    fn test_expression_hash_eq_access() {
        let env = TestEnv::from_spec(
            "\
            input i: Int32\n\
            output a(p: Int32) spawn with i := i + p\n\
            output b(q: Int32) spawn with i := a(q)\n\
            output c(r: Int32) spawn with i := a(r)",
        );
        let b_exp = env.hir.expression(env.hir.outputs[1].expr_id);
        let c_exp = env.hir.expression(env.hir.outputs[2].expr_id);
        assert!(b_exp.value_neq_ignore_parameters(c_exp));
        assert!(b_exp.value_eq(c_exp, env.ctx.as_ref()));

        let b_hash_expr = HashableExpression {
            context: env.ctx.clone(),
            expression: b_exp.clone(),
        };

        let c_hash_expr = HashableExpression {
            context: env.ctx,
            expression: c_exp.clone(),
        };
        let mut hasher_b = RandomState::new().build_hasher();
        let mut hasher_c = RandomState::new().build_hasher();
        assert_eq!(b_hash_expr, c_hash_expr);
        assert_eq!(b_hash_expr.hash(&mut hasher_b), c_hash_expr.hash(&mut hasher_c));
    }

    #[test]
    fn test_expression_type_parsing() {
        let env = TestEnv::from_spec(
            "\
            input i1: Bool\n\
            input i2: Bool\n\
            input i3: Bool\n\
            input i4: Bool\n\
            input i5: Bool\n\
            output a := i1 && i2 && i3 && i4 && i5\n\
            output b := i1 || i2 || i3 || i4 || i5\n\
            output c := i1 && !i2 && i3\n\
            output d := i1 && i2 || i3\n\
            output e := i1 && !(i2 && i3)",
        );
        let a_exp = env.hir.expression(env.hir.outputs[0].expr_id);
        let b_exp = env.hir.expression(env.hir.outputs[1].expr_id);
        let c_exp = env.hir.expression(env.hir.outputs[2].expr_id);
        let d_exp = env.hir.expression(env.hir.outputs[3].expr_id);
        let e_exp = env.hir.expression(env.hir.outputs[4].expr_id);


        assert!(matches!(AbstractExpressionType::from_expression(a_exp, env.ctx.clone()), AbstractExpressionType::Conjunction(_)));
        assert!(matches!(AbstractExpressionType::from_expression(b_exp, env.ctx.clone()), AbstractExpressionType::Disjunction(_)));
        assert!(matches!(AbstractExpressionType::from_expression(c_exp, env.ctx.clone()), AbstractExpressionType::Conjunction(_)));
        assert!(matches!(AbstractExpressionType::from_expression(d_exp, env.ctx.clone()), AbstractExpressionType::Mixed(_)));
        assert!(matches!(AbstractExpressionType::from_expression(e_exp, env.ctx), AbstractExpressionType::Mixed(_)));
    }
}
