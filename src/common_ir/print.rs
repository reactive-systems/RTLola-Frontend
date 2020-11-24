use super::*;
use crate::hir::AnnotatedType;
use itertools::Itertools;
use std::fmt::{Display, Formatter, Result};

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::PastDiscreteOffset(u) => write!(f, "{}", u),
            _ => unimplemented!(),
        }
    }
}

impl Display for WindowReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            WindowReference::SlidingRef(u) => write!(f, "SlidingWin({})", u),
            WindowReference::DiscreteRef(u) => write!(f, "DiscreteWin({})", u),
        }
    }
}

impl Display for StreamReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            StreamReference::OutRef(ix) => write!(f, "Out({})", ix),
            StreamReference::InRef(ix) => write!(f, "In({})", ix),
        }
    }
}

impl Display for AnnotatedType {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use AnnotatedType::*;
        match self {
            Int(s) => write!(f, "Int{}", s),
            Float(s) => write!(f, "Float{}", s),
            UInt(s) => write!(f, "UInt{}", s),
            Bool => write!(f, "Bool"),
            String => write!(f, "String"),
            Bytes => write!(f, "Bytes"),
            Option(t) => write!(f, "Option<{}>", t),
            Tuple(tys) => write!(f, "({})", tys.iter().map(|t| format!("{}", t)).join(",")),
            //Used in function declaration
            Numeric => write!(f, "Numeric"),
            Param(idx, name) => write!(f, "FunctionParam({}, {})", *idx, name),
        }
    }
}
