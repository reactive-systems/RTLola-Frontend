use crate::{common_ir::SRef, hir::expression::ExprId, tyc::pacing_types::ConcretePacingType, tyc::rtltc::TypeTable};

use super::Typed;

pub(crate) trait TypeChecked {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

impl TypeChecked for TypeTable {
    fn stream_type(&self, sr: SRef) -> HirType {
        self.get_type_for_stream(sr)
    }
    fn is_periodic(&self, sr: SRef) -> bool {
        matches!(self.get_type_for_stream(sr).pacing_ty, ConcretePacingType::FixedPeriodic(_))
    }
    fn is_event(&self, sr: SRef) -> bool {
        matches!(self.get_type_for_stream(sr).pacing_ty, ConcretePacingType::Event(_))
    }
    fn expr_type(&self, eid: ExprId) -> HirType {
        self.get_type_for_expr(eid)
    }
}

pub(crate) trait TypedWrapper {
    type InnerT: TypeChecked;
    fn inner_typed(&self) -> &Self::InnerT;
}

impl TypedWrapper for Typed {
    type InnerT = TypeTable;
    fn inner_typed(&self) -> &Self::InnerT {
        &self.tts
    }
}

impl<A: TypedWrapper<InnerT = T>, T: TypeChecked + 'static> TypeChecked for A {
    fn stream_type(&self, sr: SRef) -> HirType {
        self.inner_typed().stream_type(sr)
    }
    fn is_periodic(&self, sr: SRef) -> bool {
        self.inner_typed().is_periodic(sr)
    }
    fn is_event(&self, sr: SRef) -> bool {
        self.inner_typed().is_event(sr)
    }
    fn expr_type(&self, eid: ExprId) -> HirType {
        self.inner_typed().expr_type(eid)
    }
}

pub(crate) type HirType = crate::tyc::rtltc::StreamType;
/*
#[derive(Debug, Clone)]
pub(crate) struct HirType {} // TBD
*/
