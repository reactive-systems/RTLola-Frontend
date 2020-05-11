use super::*;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::frequency::hertz;
use rusttyc::Abstract;
use front::parse::NodeId;

#[derive(Debug, Clone)]
pub enum ActivationCondition<Var: Eq + Clone> {
    Conjunction(Vec<Self>),
    Disjunction(Vec<Self>),
    Stream(Var),
    True,
}

impl <Var:Eq + Clone> ActivationCondition<Var> {
    fn flatten(&mut self){
        use ActivationCondition::*;
        match self{
            True | Stream(_) => return,
            Conjunction(childs) => {
                childs.iter_mut().for_each(ActivationCondition::flatten);
                let mut flattend = Vec::with_capacity(childs.len());
                childs.iter_mut().for_each(|child|{
                    match child {
                        Conjunction(r) => flattend.append(r),
                        x => flattend.push(x.clone()),
                    }
                });
                *childs = flattend;
            },
            Disjunction(childs) => {
                childs.iter_mut().for_each(ActivationCondition::flatten);
                let mut flattend = Vec::with_capacity(childs.len());
                childs.iter_mut().for_each(|child|{
                    match child {
                        Disjunction(r) => flattend.append(r),
                        x => flattend.push(x.clone()),
                    }
                });
                *childs = flattend;
            }
        }
    }
}

impl <Var: Eq+Clone> PartialEq for ActivationCondition<Var> {
    fn eq(&self, other: &Self) -> bool{
        true // Todo: Implement me
    }
}
impl <Var: Eq+Clone> Eq for ActivationCondition<Var> {}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Freq(pub(crate) UOM_Frequency);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    MixedEventPeriodic(ActivationCondition<NodeId>, Freq),
    IncompatibleFrequencies(Freq, Freq),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractPacingType{
    /// An event stream is extended when its activation condition is satisfied.
    Event(ActivationCondition<NodeId>),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Any
}

#[derive(Copy, Clone, Eq, PartialEq)]
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
            (Event(ac), Periodic(f)) | (Periodic(f), Event(ac)) => Err(UnificationError::MixedEventPeriodic(ac, f)),
            (Event(ac1), Event(ac2)) => {
                use ActivationCondition::*;
                let res = match (ac1, ac2) {
                    (True, x) | (x, True)=> x,
                    (Stream(x), Stream(y)) => Conjunction(vec![Stream(x),Stream(y)]),
                    (Stream(x), Conjunction(other)) | (Conjunction(other), Stream(x)) => {
                        let mut new_ac = other.clone();
                        new_ac.push(Stream(x));
                        Conjunction(new_ac)
                    },
                    (Conjunction(lhs), Conjunction(rhs)) => {
                        let mut new_ac = lhs.clone();
                        new_ac.append(&mut rhs.clone());
                        Conjunction(new_ac)
                    },
                    _ => return Err(UnificationError::Other("Unimplemented!".into()))
                    //output a @ i1
                    //output c @ 1 Hz
                    // output b @ c = a + c
                };
                Ok(Event(res))
            },
            _ => Err(UnificationError::Other("Unimplemented!".into()))
        }
    }
}

#[test]
fn test_ac_flatten() {
    let mut conjs = vec![ActivationCondition::Stream(42); 5];
    let disj = vec![ActivationCondition::Stream(7); 5];

    let child_a = ActivationCondition::Conjunction(conjs.clone());
    let child_b = ActivationCondition::Disjunction(disj.clone());
    let mut ac = ActivationCondition::Conjunction(vec![child_a.clone(), child_b.clone()]);

    conjs.push(child_b);
    let expected = ActivationCondition::Conjunction(conjs);

    ac.flatten();

    assert_eq!(ac, expected);
}