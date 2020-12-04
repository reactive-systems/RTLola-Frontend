use crate::{common_ir::SRef, hir::expression::ExprId, hir::modes::TypeTables};

use super::Typed;

pub(crate) trait TypeChecked {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

impl TypeChecked for TypeTables {
    fn stream_type(&self, _sr: SRef) -> HirType {
        unimplemented!()
    }
    fn is_periodic(&self, _sr: SRef) -> bool {
        todo!()
    }
    fn is_event(&self, _sr: SRef) -> bool {
        todo!()
    }
    fn expr_type(&self, _eid: ExprId) -> HirType {
        unimplemented!()
    }
}

pub(crate) trait TypedWrapper {
    type InnerT: TypeChecked;
    fn inner_typed(&self) -> &Self::InnerT;
}

impl TypedWrapper for Typed {
    type InnerT = TypeTables;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.tts
    }
}

impl<A: TypedWrapper<InnerT = T>, T: TypeChecked + 'static> TypeChecked for A {
    fn stream_type(&self, _sr: SRef) -> HirType {
        unimplemented!()
    }
    fn is_periodic(&self, _sr: SRef) -> bool {
        todo!()
    }
    fn is_event(&self, _sr: SRef) -> bool {
        todo!()
    }
    fn expr_type(&self, _eid: ExprId) -> HirType {
        unimplemented!()
    }
}
#[derive(Debug, Clone)]
pub(crate) struct HirType {} // TBD
