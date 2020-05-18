use super::*;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::frequency::hertz;
use num::{CheckedDiv, Integer};
use rusttyc::Abstract;
use front::parse::{NodeId, Span};
use front::ast::{Expression, ExpressionKind, LitKind, BinOp, UnOp};
use biodivine_lib_bdd::{Bdd, BddVariableSet};
use std::convert::TryFrom;
use uom::num_rational::Ratio;

type ActivationCondition = Bdd;


pub fn parse_ac(ast_expr: Option<Expression>, var_set:&BddVariableSet)->Result<ActivationCondition, (String, Span)>{
    if ast_expr.is_none()
    {
        return Ok(var_set.mk_true());
    }
    let ast_expr = ast_expr.unwrap();
    use ExpressionKind::*;
    match ast_expr.kind {
        Lit(l) => match l.kind {
            LitKind::Bool(b) => {
                if b {
                    Ok(var_set.mk_true())
                }else{
                    Ok(var_set.mk_false())
                }
            }
            _ => Err(("Unexpected literal in activation condition".into(), l.span))
        },
        Ident(i) => Ok(var_set.mk_var_by_name(i.name.as_str())),
        Binary(op, l, r) => {
            let ac_l = parse_ac(Some(*l), var_set)?;
            let ac_r = parse_ac(Some(*r), var_set)?;
            use BinOp::*;
            match op {
                And => Ok(ac_l.and(&ac_r)),
                Or => Ok(ac_l.or(&ac_r)),
                Eq => Ok(ac_l.iff(&ac_r)),
                _ => Err(("Unexpected binary operator in activation condition".into(), ast_expr.span))
            }
        },
        Unary(op, l) => {
            let ac_l = parse_ac(Some(*l), var_set)?;
            match op {
                UnOp::Not => Ok(ac_l.not()),
                _ => Err(("Unexpected unary operator in activation condition".into(), ast_expr.span))
            }
        },
        _ => Err(("Unexpected expression in activation condition".into(), ast_expr.span)),
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
                freq.clone().into_format_args(uom::si::frequency::hertz, uom::fmt::DisplayStyle::Abbreviation)
            ),
        }
    }
}

impl Freq {
    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, UnificationError> {
        let lhs = match self {
            Freq::Fixed(f) => f,
            Any => return Ok(false),
        };
        let rhs = match other {
            Freq::Fixed(f) => f,
            Any => return Ok(false),
        };

        if lhs.get::<hertz>() < rhs.get::<hertz>() {
            return Ok(false);

        }
        match lhs.get::<hertz>().checked_div(&rhs.get::<hertz>()) {
            Some(q) => Ok(q.is_integer()),
            None => Err(UnificationError::Other(format!("division of frequencies `{:?}`/`{:?}` failed", &lhs, &rhs))),
        }
    }

    pub(crate) fn conjunction(&self, other: &Freq) -> Freq {
        let (numer_left, denom_left) = match self {
            Freq::Any => return other.clone(),
            Freq::Fixed(f) => (f.get::<hertz>().numer().clone(), f.get::<hertz>().denom().clone()),
        };
        let (numer_right, denom_right) = match other {
            Freq::Any => return self.clone(),
            Freq::Fixed(f) => (f.get::<hertz>().numer().clone(), f.get::<hertz>().denom().clone()),
        };
        // gcd(self, other) = gcd(numer_left, numer_right) / lcm(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `Rational`
        let r1: i64 = i64::try_from(numer_left.gcd(&numer_right)).unwrap();
        let r2: i64 = i64::try_from(denom_left.gcd(&denom_right)).unwrap();
        let r : Ratio<i64> = Ratio::new(r1, r2);
        Freq::Fixed(UOM_Frequency::new::<hertz>(r))
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    MixedEventPeriodic(ActivationCondition, Freq),
    IncompatibleFrequencies(Freq, Freq),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractPacingType{
    /// An event stream is extended when its activation condition is satisfied.
    Event(ActivationCondition),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursivePacingType {
    Other
}

impl rusttyc::TypeVariant for RecursivePacingType {
    fn arity(self) -> u8 {
        0
    }
}

impl rusttyc::Abstract for AbstractPacingType {
    type Error = UnificationError;
    type Variant = RecursivePacingType;

    fn unconstrained() -> Self {
        AbstractPacingType::Any
    }

    fn meet(self, other: Self) -> Result<Self, Self::Error> {
        use AbstractPacingType::*;
        match (self, other) {
            (Any, x) | (x, Any) => Ok(x),
            (Event(ac), Periodic(f)) | (Periodic(f), Event(ac)) => Err(UnificationError::MixedEventPeriodic(ac, f)),
            (Event(ac1), Event(ac2)) => {
                Ok(Event(ac1.and(&ac2)))
            },
            (Periodic(f1), Periodic(f2)) => {
                if f1.is_multiple_of(&f2)? || f2.is_multiple_of(&f1)? {
                    Ok(Periodic(f1.conjunction(&f2)))
                } else {
                    Err(UnificationError::IncompatibleFrequencies(f1, f2))
                }
            }
        }
    }
}