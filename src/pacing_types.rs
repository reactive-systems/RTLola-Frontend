use super::*;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::frequency::hertz;
use rusttyc::Abstract;

pub enum ActivationCondition<Var> {
    Conjunction(Vec<Self>),
    Disjunction(Vec<Self>),
    Stream(Var),
    True,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Freq(pub(crate) UOM_Frequency);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    MixedEventPeriodic(ActivationCondition<NodeId>, Freq),
    IncompatibleFrequencies(Freq, Freq),
    IncompatibleActivationConditions(ActivationCondition<NodeId>, ActivationCondition<NodeId>),
    Other(String),
}

pub enum AbstractPacingType{
    /// An event stream is extended when its activation condition is satisfied.
    Event(ActivationCondition<NodeId>),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any
}

impl rusttyc::Abstract for AbstractPacingType {
    type Error = UnificationError;
    type Variant = ();

    fn unconstrained() -> Self {
        AbstractPacingType::Any
    }

    fn meet(self, other: Self) -> Result<Self, Self::Error> {
        use AbstractPacingType::*;
        match (self, other) {
            (Event(ac), Periodic(f)) | (Periodic(f), Event(ac)) => Err(UnificationError::MixedEventPeriodic(ac, f)),
            (Event(ac1), Event(ac2)) => {
                use ActivationCondition::*;
                let res = match (ac1, ac2) {
                    (True, x) | (x, True)=> x,
                    (Stream(x), Stream(y)) => Conjunction(vec![x,y]),
                    (Stream(x), Conjunction(other)) | (Conjunction(other), Stream(x)) => {
                        let mut new_ac = other.clone();
                        new_ac.push(x);
                        Conjunction(new_ac)
                    },
                    (Conjunction(lhs), Conjunction(rhs)) => {
                        let mut new_ac = lhs.clone();
                        new_ac.append(&rhs.clone());
                        Conjunction(new_ac)
                    },
                    //output a @ i1
                    //output c @ 1 Hz
                    // output b @ c = a + c
                };
            },
        }
    }
}