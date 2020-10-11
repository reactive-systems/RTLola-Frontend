#![allow(dead_code)]

use crate::{hir::modes::HirMode, RTLolaHIR};
use std::cmp::Ordering;

pub(crate) mod sccp; //sparse conditional constant propagation

/// Transforms an intermediate representation to an optimized one.
/// Currently Sparse Conditional Constant Propagation is implemented
pub(crate) trait Transformation<M: HirMode> {
    fn transform(ir: RTLolaHIR<M>) -> RTLolaHIR<M>;
}

/// Abstract lattice values used for the transfromations
#[derive(Debug, Clone, PartialEq)]
enum LatticeValues<A: PartialEq> {
    Bot,
    Val(A),
    Top,
}

impl<A: PartialEq> PartialOrd for LatticeValues<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (LatticeValues::Top, LatticeValues::Top) => Some(Ordering::Equal),
            (LatticeValues::Top, _) => Some(Ordering::Greater),
            (LatticeValues::Val(_), LatticeValues::Top) => Some(Ordering::Less),
            (LatticeValues::Val(a), LatticeValues::Val(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (LatticeValues::Val(_), LatticeValues::Bot) => Some(Ordering::Greater),
            (LatticeValues::Bot, LatticeValues::Bot) => Some(Ordering::Equal),
            (LatticeValues::Bot, _) => Some(Ordering::Less),
        }
    }
}

impl<A: PartialEq> LatticeValues<A> {
    fn meet(lhs: LatticeValues<A>, rhs: LatticeValues<A>) -> LatticeValues<A> {
        match lhs.partial_cmp(&rhs) {
            None => LatticeValues::Bot,
            Some(Ordering::Less) => lhs,
            Some(Ordering::Equal) => rhs,
            Some(Ordering::Greater) => rhs,
        }
    }
    fn join(lhs: LatticeValues<A>, rhs: LatticeValues<A>) -> LatticeValues<A> {
        match lhs.partial_cmp(&rhs) {
            None => LatticeValues::Top,
            Some(Ordering::Less) => rhs,
            Some(Ordering::Equal) => lhs,
            Some(Ordering::Greater) => lhs,
        }
    }
}

#[cfg(test)]
mod lattice_tests {
    use super::LatticeValues;

    #[test]
    fn join_test() {
        let top = LatticeValues::<i8>::Top;
        let bot = LatticeValues::<i8>::Bot;
        let v1 = LatticeValues::<i8>::Val(6);
        let v2 = LatticeValues::<i8>::Val(6);
        let v3 = LatticeValues::<i8>::Val(8);

        assert_eq!(LatticeValues::join(top.clone(), top.clone()), top);
        assert_eq!(LatticeValues::join(v1.clone(), top.clone()), top);
        assert_eq!(LatticeValues::join(v1.clone(), v2.clone()), v1);
        assert_eq!(LatticeValues::join(v1.clone(), v3.clone()), top);
        assert_eq!(LatticeValues::join(v3.clone(), bot.clone()), v3);
    }

    #[test]
    fn meet_test() {
        let top = LatticeValues::<i8>::Top;
        let bot = LatticeValues::<i8>::Bot;
        let v1 = LatticeValues::<i8>::Val(6);
        let v2 = LatticeValues::<i8>::Val(6);
        let v3 = LatticeValues::<i8>::Val(8);

        assert_eq!(LatticeValues::meet(top.clone(), top.clone()), top);
        assert_eq!(LatticeValues::meet(v1.clone(), top.clone()), v1);
        assert_eq!(LatticeValues::meet(v1.clone(), v2.clone()), v1);
        assert_eq!(LatticeValues::meet(v1.clone(), v3.clone()), bot);
        assert_eq!(LatticeValues::meet(v3.clone(), bot.clone()), bot);
    }
}
