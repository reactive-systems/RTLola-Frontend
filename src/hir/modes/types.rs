use crate::hir::Hir;
use crate::{common_ir::SRef, hir::expression::ExprId, tyc::pacing_types::ConcretePacingType};

use super::{EvaluationOrder, OrderedMode, TypedMode, TypedTrait};

impl Hir<TypedMode> {
    pub(crate) fn build_evaluation_order(self) -> Hir<OrderedMode> {
        let order = EvaluationOrder::analyze(&self);

        let old_mode = self.mode.clone();
        let mode = OrderedMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: old_mode,
            layers: order,
        };

        Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            mode,
        }
    }
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
