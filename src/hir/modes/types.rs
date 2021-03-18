use crate::{common_ir::SRef, hir::expression::ExprId, tyc::pacing_types::ConcretePacingType};

use super::TypedMode;

pub(crate) trait TypedTrait {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
}

impl TypedTrait for TypedMode {
    fn stream_type(&self, sr: SRef) -> HirType {
        self.tts.get_type_for_stream(sr)
    }
    fn is_periodic(&self, sr: SRef) -> bool {
        matches!(self.tts.get_type_for_stream(sr).pacing_ty, ConcretePacingType::FixedPeriodic(_))
    }
    fn is_event(&self, sr: SRef) -> bool {
        matches!(self.tts.get_type_for_stream(sr).pacing_ty, ConcretePacingType::Event(_))
    }
    fn expr_type(&self, eid: ExprId) -> HirType {
        self.tts.get_type_for_expr(eid)
    }
}

pub(crate) type HirType = crate::tyc::rtltc::StreamType;
/*
#[derive(Debug, Clone)]
pub(crate) struct HirType {} // TBD
*/
