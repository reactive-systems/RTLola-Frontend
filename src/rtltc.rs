use super::*;

use crate::astclimb::Variable;
use crate::types::IAbstractType;
use front::analysis::naming::DeclarationTable;
use front::ast::LolaSpec;
use rusttyc::{Abstract, TypeChecker};

#[derive()]
pub struct LolaTypChecker<'a> {
    pub(crate) ast: LolaSpec,
    pub(crate) declarations: DeclarationTable<'a>,
}

impl<'a> LolaTypChecker<'a> {
    pub fn new(spec: &LolaSpec, declarations: DeclarationTable<'a>) -> Self {
        LolaTypChecker {
            ast: spec.clone(),
            declarations: declarations.clone(),
        }
    }

    pub fn generate_raw_table(&self) -> Vec<(i32, front::ty::Ty)> {
        vec![]
    }
}
