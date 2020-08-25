use biodivine_lib_bdd::boolean_expression::BooleanExpression;
use biodivine_lib_bdd::{Bdd, BddVariableSet};
use front::analysis::naming::{Declaration, DeclarationTable};
use front::ast::{BinOp, Expression, ExpressionKind, LitKind};
use front::parse::{NodeId, Span};
use front::reporting::{Handler, LabeledSpan};
use num::{CheckedDiv, Integer};
use rusttyc::{TcErr, TcKey};
use std::convert::TryFrom;
use uom::lib::collections::HashMap;
use uom::num_rational::Ratio;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;

/// Parses either a periodic or an event based pacing type from an expression
pub fn parse_abstract_type<'a>(
    ast_expr: &Expression,
    var_set: &BddVariableSet,
    decl: &'a DeclarationTable,
) -> Result<AbstractPacingType, (String, Span)> {
    match &ast_expr.kind {
        ExpressionKind::Lit(l) => match l.kind {
            LitKind::Bool(_) => {
                let ac = parse_ac(ast_expr, var_set, decl)?;
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
            let ac = parse_ac(ast_expr, var_set, decl)?;
            Ok(AbstractPacingType::Event(ac))
        }
    }
}

fn parse_ac<'a>(
    ast_expr: &Expression,
    var_set: &BddVariableSet,
    decl: &'a DeclarationTable,
) -> Result<Bdd, (String, Span)> {
    use ExpressionKind::*;
    match &ast_expr.kind {
        Lit(l) => Err((
            "Literals are not allowed in activation conditions.".into(),
            l.span,
        )),
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
        Binary(op, l, r) => {
            let ac_l = parse_ac(l, var_set, decl)?;
            let ac_r = parse_ac(r, var_set, decl)?;
            use BinOp::*;
            match op {
                And => Ok(ac_l.and(&ac_r)),
                Or => Ok(ac_l.or(&ac_r)),
                _ => Err((
                    "Only '&' (and) or '|' (or) are allowed in activation conditions.".into(),
                    ast_expr.span,
                )),
            }
        }
        ParenthesizedExpression(_, exp, _) => parse_ac(exp, var_set, decl),
        _ => Err((
            "An activation condition can only contain literals and binary operators.".into(),
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
    MixedEventPeriodic(AbstractPacingType, AbstractPacingType),
    IncompatibleFrequencies(AbstractPacingType, AbstractPacingType),
    Other(String),
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
            let id = NodeId::new(s.parse::<usize>().unwrap());
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
            IncompatibleFrequencies(f1, f2) => format!(
                "Found incompatible frequencies: '{}' and '{}'",
                f1.to_string(bdd_vars, input_name),
                f2.to_string(bdd_vars, input_name)
            ),
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
            let span = LabeledSpan::new(spans[k1], "here", true);
            handler.error_with_span(msg, span);
        }
        TcErr::Bound(k1, k2, err) => {
            let msg = &format!("In pacing type analysis:\n {}", err.to_string(vars, names));
            let span = LabeledSpan::new(spans[k1], "here", true);
            handler.error_with_span(msg, span);
        }
        TcErr::ChildAccessOutOfBound(key, ty, idx) => {
            let msg = &format!(
                "Child type out of bounds for type: {}",
                ty.to_string(vars, names)
            );
            let span = LabeledSpan::new(spans[key], "here", true);
            handler.error_with_span(msg, span);
        }
        //TODO
        TcErr::ExactTypeViolation(_, _) => {}
        TcErr::ConflictingExactBounds(_, _, _) => {}
    }
}

// Abstract Type Definition
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractPacingType {
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
            (Event(ac), Periodic(f)) => Err(UnificationError::MixedEventPeriodic(
                Event(ac.clone()),
                Periodic(f.clone()),
            )),
            (Periodic(f), Event(ac)) => Err(UnificationError::MixedEventPeriodic(
                Periodic(f.clone()),
                Event(ac.clone()),
            )),
            (Event(ac1), Event(ac2)) => Ok(Event(ac1.and(&ac2))),
            (Periodic(f1), Periodic(f2)) => {
                if let Freq::Any = f1 {
                    Ok(Periodic(f2.clone()))
                } else if let Freq::Any = f2 {
                    Ok(Periodic(f1.clone()))
                } else if f1.is_multiple_of(&f2)? || f2.is_multiple_of(&f1)? {
                    Ok(Periodic(f1.conjunction(&f2)))
                } else {
                    Err(UnificationError::IncompatibleFrequencies(
                        Periodic(f1.clone()),
                        Periodic(f2.clone()),
                    ))
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
    pub(crate) fn from_expression(exp: BooleanExpression) -> Result<Self, String> {
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
                let id = s.parse::<usize>();
                match id {
                    Ok(i) => Ok(ActivationCondition::Stream(NodeId::new(i))),
                    Err(_) => Err("Wrong Variable in AC".to_string()),
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
            _ => Err("Unsupported Operation in AC".to_string()),
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
    ) -> Result<Self, String> {
        match abs_t {
            AbstractPacingType::Any => Ok(ConcretePacingType::Constant),
            AbstractPacingType::Event(b) => {
                ActivationCondition::from_expression(b.to_boolean_expression(vars))
                    .map(|ac| ConcretePacingType::Event(ac))
            }
            AbstractPacingType::Periodic(freq) => match freq {
                Freq::Fixed(f) => Ok(ConcretePacingType::FixedPeriodic(f.clone())),
                Freq::Any => Ok(ConcretePacingType::Periodic),
            },
        }
    }
}
