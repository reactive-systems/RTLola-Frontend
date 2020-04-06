use super::*;

use crate::types::IAbstractType;
use crate::astclimb::Variable;
use front::ast::LolaSpec;
use rusttyc::{Abstract, TypeChecker};

#[derive()]
pub struct TypChecker {
    pub(crate) ast: LolaSpec,
}

impl TypChecker {
    pub fn new(spec: &LolaSpec) -> Self {
        TypChecker { ast: spec.clone() }
    }

    pub fn generate_raw_table(&self) -> Vec<(i32, front::ty::Ty)> {
        vec![]
    }
}
