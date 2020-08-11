use biodivine_lib_bdd::{Bdd, BddVariableSet};
use biodivine_lib_bdd::boolean_expression::BooleanExpression;
use front::ast::{BinOp, Expression, ExpressionKind, LitKind, UnOp};
use front::parse::NodeId;
use front::parse::Span;
use num::{CheckedDiv, Integer};
use std::convert::TryFrom;
use uom::num_rational::Ratio;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;
use std::collections::HashMap;


/// Parses either a periodic or an event based pacing type from an expression
pub fn parse_abstract_type(
    ast_expr: Option<&Expression>,
    var_set: &BddVariableSet,
) -> Result<AbstractPacingType, (String, Span)> {
    if ast_expr.is_none() {
        return Ok(AbstractPacingType::Event(var_set.mk_true()));
    }
    let ast_expr = ast_expr.unwrap();
    match &ast_expr.kind {
        ExpressionKind::Lit(l) => match l.kind {
            LitKind::Bool(_) => {
                let ac = parse_ac(ast_expr, var_set)?;
                Ok(AbstractPacingType::Event(ac))
            }
            _ => {
                let freq: UOM_Frequency = ast_expr
                    .parse_freqspec()
                    .map_err(|reason| (reason, ast_expr.span))?;
                Ok(AbstractPacingType::Periodic(Freq::Fixed(freq)))
            }
        },
        _ => {
            let ac = parse_ac(ast_expr, var_set)?;
            Ok(AbstractPacingType::Event(ac))
        }
    }
}

fn parse_ac(
    ast_expr: &Expression,
    var_set: &BddVariableSet,
) -> Result<Bdd, (String, Span)> {
    use ExpressionKind::*;
    match &ast_expr.kind {
        Lit(l) => match l.kind {
            LitKind::Bool(b) => {
                if b {
                    Ok(var_set.mk_true())
                } else {
                    Ok(var_set.mk_false())
                }
            }
            _ => Err(("Unexpected literal in activation condition".into(), l.span)),
        },
        Ident(i) => Ok(var_set.mk_var_by_name(i.name.as_str())),
        Binary(op, l, r) => {
            let ac_l = parse_ac(l, var_set)?;
            let ac_r = parse_ac(r, var_set)?;
            use BinOp::*;
            match op {
                And => Ok(ac_l.and(&ac_r)),
                Or => Ok(ac_l.or(&ac_r)),
                Eq => Ok(ac_l.iff(&ac_r)),
                _ => Err((
                    "Unexpected binary operator in activation condition".into(),
                    ast_expr.span,
                )),
            }
        }
        _ => Err((
            "Unexpected expression in activation condition".into(),
            ast_expr.span,
        )),
    }
}

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
            Freq::Any => return other.clone(),
            Freq::Fixed(f) => (
                f.get::<hertz>().numer().clone(),
                f.get::<hertz>().denom().clone(),
            ),
        };
        let (numer_right, denom_right) = match other {
            Freq::Any => return self.clone(),
            Freq::Fixed(f) => (
                f.get::<hertz>().numer().clone(),
                f.get::<hertz>().denom().clone(),
            ),
        };
        // gcd(self, other) = gcd(numer_left, numer_right) / lcm(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `Rational`
        let r1: i64 = i64::try_from(numer_left.gcd(&numer_right)).unwrap();
        let r2: i64 = i64::try_from(denom_left.gcd(&denom_right)).unwrap();
        let r: Ratio<i64> = Ratio::new(r1, r2);
        Freq::Fixed(UOM_Frequency::new::<hertz>(r))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    MixedEventPeriodic(Bdd, Freq),
    IncompatibleFrequencies(Freq, Freq),
    Other(String),
}

// Abstract Type Definition
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractPacingType{
    /// An event stream is extended when its activation condition is satisfied.
    Event(Bdd),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any,
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
            (Event(ac), Periodic(f)) | (Periodic(f), Event(ac)) => {
                Err(UnificationError::MixedEventPeriodic(ac.clone(), f.clone()))
            }
            (Event(ac1), Event(ac2)) => Ok(Event(ac1.and(&ac2))),
            (Periodic(f1), Periodic(f2)) => {
                if f1.is_multiple_of(&f2)? || f2.is_multiple_of(&f1)? {
                    Ok(Periodic(f1.conjunction(&f2)))
                } else {
                    Err(UnificationError::IncompatibleFrequencies(f1.clone(), f2.clone()))
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

    fn with_children<I>(&self, _children: I) -> Self where
        I: IntoIterator<Item=Self> {
        unreachable!()
    }
}

impl AbstractPacingType {
    pub(crate) fn to_string(&self, vars: &BddVariableSet) -> String{
        match self{
            AbstractPacingType::Event(b) => {
                format!("Event({})", b.to_boolean_expression(vars))
            },
            AbstractPacingType::Periodic(freq) => {
                format!("Periodic({})", freq)
            },
            AbstractPacingType::Any => 
            {
                "Any".to_string()
            }
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
    Stream(u32),
    /**
    Whenever an event-based stream produces a new value.
    */
    True,
}

/*
impl ActivationCondition {
    pub(crate) fn from_expression(exp: BooleanExpression) -> Result<Self, String>{
        use BooleanExpression::*;
        match exp {
            Const(b) => {
                if b {
                    Ok(ActivationCondition::True)
                } else {
                    Err("False in Activation Condition".to_string())
                }
            }
            Variable(s) => {
                let id = s.parse::<u32>();
                match id{
                    Ok(i) => Ok(ActivationCondition::Stream(i)),
                    Err(_) => Err("Wrong Variable in AC".to_string())
                }
            }
            And(left, right) => {
                let l = ActivationCondition::from_expression(*left)?;
                let r = ActivationCondition::from_expression(*right)?;
                match (l, r) {
                    (ActivationCondition::Conjunction(mut left), ActivationCondition::Conjunction(mut right)) => {
                        left.append(&mut right);
                        Ok(ActivationCondition::Conjunction(left))
                    },
                    (ActivationCondition::Conjunction(mut other_con), other_ac) |
                    (other_ac, ActivationCondition::Conjunction(mut other_con)) => {
                        other_con.push(other_ac);
                        Ok(ActivationCondition::Conjunction(other_con))
                    },

                }
            }
        }
    }
}
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum ConcretePacingType {
    Event(ActivationCondition),
    Periodic(UOM_Frequency)
}