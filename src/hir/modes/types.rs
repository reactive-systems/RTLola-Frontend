use crate::{common_ir::SRef, hir::expression::ExprId};

use super::Typed;

pub(crate) trait TypeChecked {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

impl TypeChecked for Typed {
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

pub(crate) struct HirType {} // TBD
