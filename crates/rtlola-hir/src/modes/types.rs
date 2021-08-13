use super::{Typed, TypedTrait};
use crate::hir::{ExprId, SRef};
use crate::type_check::{ConcreteValueType, StreamType};

impl TypedTrait for Typed {
    fn stream_type(&self, sr: SRef) -> HirType {
        self.get_type_for_stream(sr)
    }

    fn is_periodic(&self, sr: SRef) -> bool {
        self.get_type_for_stream(sr).pacing_ty.is_periodic()
    }

    fn is_event(&self, sr: SRef) -> bool {
        self.get_type_for_stream(sr).pacing_ty.is_event_based()
    }

    fn expr_type(&self, eid: ExprId) -> HirType {
        self.get_type_for_expr(eid)
    }

    fn get_parameter_type(&self, sr: SRef, idx: usize) -> ConcreteValueType {
        self.param_types[&(sr, idx)].clone()
    }
}

impl Typed {
    /// Returns for a given StreamReference the corresponding [StreamType].
    pub fn get_type_for_stream(&self, sref: SRef) -> StreamType {
        self.stream_types[&sref].clone()
    }

    /// Returns for a given [Id of an expression](ExprId) the corresponding [StreamType].
    pub fn get_type_for_expr(&self, exprid: ExprId) -> StreamType {
        self.expression_types[&exprid].clone()
    }
}

/// Represents the type of a stream in the [RtLolaHir]
pub(crate) type HirType = StreamType;
