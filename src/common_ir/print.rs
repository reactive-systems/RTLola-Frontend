use super::*;
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
        write!(f, "Win({})", self.0)
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
