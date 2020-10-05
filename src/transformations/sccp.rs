use crate::{hir::expression::Constant, transformations::Transformation};
use crate::{hir::modes::Complete, RTLolaHIR};
use std::cmp::Ordering;

pub(crate) struct SCCP;
impl Transformation<Complete> for SCCP {
    fn transform(_ir: RTLolaHIR<Complete>) -> RTLolaHIR<Complete> {
        // transform concrete stream expressions to lattice expressions.
        unimplemented!()
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Constant::Str(s1), Constant::Str(s2)) => s1.partial_cmp(s2),
            (Constant::Bool(b1), Constant::Bool(b2)) => b1.partial_cmp(b2),
            (Constant::UInt(u1), Constant::UInt(u2)) => u1.partial_cmp(u2),
            (Constant::Int(i1), Constant::Int(i2)) => i1.partial_cmp(i2),
            (Constant::Float(f1), Constant::Float(f2)) => f1.partial_cmp(f2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod sccp_tests {
    use super::LatticeValues;
    use crate::hir::RTLolaHIR;
    use crate::hir::{Constant, FullInformationHirMode};
    use crate::transformations::Transformation;
    use crate::FrontendConfig;

    fn spec_to_ir(spec: &str) -> RTLolaHIR<FullInformationHirMode> {
        crate::parse_to_hir("stdin", spec, FrontendConfig::default()).expect("spec was invalid")
    }

    fn transform(ir: RTLolaHIR<FullInformationHirMode>) -> RTLolaHIR<FullInformationHirMode> {
        crate::transformations::sccp::SCCP::transform(ir)
    }

    #[test]
    fn partcial_cmp_test() {
        let top = LatticeValues::<Constant>::Top;
        let bot = LatticeValues::<Constant>::Bot;
        let v1 = LatticeValues::<Constant>::Val(Constant::Int(6));
        let v2 = LatticeValues::<Constant>::Val(Constant::Int(6));
        let v3 = LatticeValues::<Constant>::Val(Constant::Int(8));

        assert_eq!(top == bot, false);
        assert_eq!(top <= bot, false);
        assert_eq!(top < bot, false);
        assert_eq!(top >= bot, true);
        assert_eq!(top > bot, true);
        assert_eq!(top == v1, false);
        assert_eq!(top <= v1, false);
        assert_eq!(top < v1, false);
        assert_eq!(top >= v1, true);
        assert_eq!(top > v1, true);
        assert_eq!(v1 == v2, true);
        assert_eq!(v1 <= v2, true);
        assert_eq!(v1 < v2, false);
        assert_eq!(v1 >= v2, true);
        assert_eq!(v1 > v2, false);
        assert_eq!(v2 == v3, false);
        assert_eq!(v2 <= v3, true);
        assert_eq!(v2 < v3, true);
        assert_eq!(v2 >= v3, false);
        assert_eq!(v2 > v3, false);
        assert_eq!(v3 == bot, false);
        assert_eq!(v3 <= bot, false);
        assert_eq!(v3 < bot, false);
        assert_eq!(v3 >= bot, true);
        assert_eq!(v3 > bot, true);
    }

    #[test]
    fn simple_sccp_test() {
        let hir = spec_to_ir("input a: Int64\noutput const: Int64 := 6\n trigger a < const");
        let _sccp = transform(hir);
    }
}
