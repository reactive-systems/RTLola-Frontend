use crate::{
    hir::ExprId,
    hir::SRef,
    type_check::{ConcretePacingType, ConcreteValueType},
};
use crate::{hir::Hir, type_check::StreamType};

use super::{Ordered, OrderedMode, Typed, TypedMode, TypedTrait};

impl Hir<TypedMode> {
    pub fn build_evaluation_order(self) -> Hir<OrderedMode> {
        let order = Ordered::analyze(&self);

        let mode = OrderedMode {
            ir_expr: self.mode.ir_expr,
            dependencies: self.mode.dependencies,
            types: self.mode.types,
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

impl TypedTrait for Typed {
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

impl Typed {
    /// For a given StreamReference, lookup the corresponding StreamType.
    pub fn get_type_for_stream(&self, sref: SRef) -> StreamType {
        self.stream_types[&sref].clone()
    }

    /// For a given Expression Id, lookup the corresponding StreamType.
    pub fn get_type_for_expr(&self, exprid: ExprId) -> StreamType {
        self.expression_types[&exprid].clone()
    }

    /// Returns the Value Type of the `idx`-th Parameter for the Stream `stream`.
    pub fn get_parameter_type(&self, stream: SRef, idx: usize) -> ConcreteValueType {
        self.param_types[&(stream, idx)].clone()
    }
}

pub(crate) type HirType = StreamType;
