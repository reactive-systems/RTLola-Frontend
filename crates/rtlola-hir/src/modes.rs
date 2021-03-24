pub(crate) mod dependencies;
pub(crate) mod ir_expr;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod types;

use self::dependencies::{DependencyErr, DependencyGraph, Streamdependencies, Windowdependencies};
use self::types::HirType;
use crate::hir::{ExprId, Hir, SRef, WRef};
use crate::modes::memory_bounds::MemorizationBound;
use crate::modes::ordering::StreamLayers;
use crate::type_check::{ConcreteValueType, StreamType};
use rtlola_reporting::Handler;
use std::collections::HashMap;

pub trait HirMode {}

pub trait HirStage: Sized {
    type Error;
    type NextStage: HirMode;
    fn progress(self, handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error>;
}

#[derive(Clone, Debug, HirMode)]
pub struct BaseMode {}

impl HirStage for Hir<BaseMode> {
    type Error = DependencyErr;
    type NextStage = DepAnaMode;
    fn progress(self, _handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let dependencies = DepAna::analyze(&self)?;
        let mode = DepAnaMode { dependencies };
        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            expr_maps: self.expr_maps,
            mode,
        })
    }
}

impl Hir<BaseMode> {
    pub fn analyze_dependencies(self, handler: &Handler) -> Result<Hir<DepAnaMode>, DependencyErr> {
        self.progress(handler)
    }
}

#[derive(Debug, Clone)]
pub struct DepAna {
    direct_accesses: Streamdependencies,
    transitive_accesses: Streamdependencies,
    direct_accessed_by: Streamdependencies,
    transitive_accessed_by: Streamdependencies,
    aggregated_by: Windowdependencies,
    aggregates: Windowdependencies,
    graph: DependencyGraph,
}

#[covers_functionality(DepAnaTrait, dependencies)]
#[derive(Debug, Clone, HirMode)]
pub struct DepAnaMode {
    dependencies: DepAna,
}

#[mode_functionality]
pub trait DepAnaTrait {
    fn direct_accesses(&self, who: SRef) -> Vec<SRef>;

    fn transitive_accesses(&self, who: SRef) -> Vec<SRef>;

    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn transitive_accessed_by(&self, who: SRef) -> Vec<SRef>;

    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    fn graph(&self) -> &DependencyGraph;
}

impl HirStage for Hir<DepAnaMode> {
    type Error = String;
    type NextStage = TypedMode;
    fn progress(self, handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let tts = crate::type_check::type_check(&self, handler)?;

        let mode = TypedMode { dependencies: self.mode.dependencies, types: tts };
        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            expr_maps: self.expr_maps,
            mode,
        })
    }
}

impl Hir<DepAnaMode> {
    pub fn check_types(self, handler: &Handler) -> Result<Hir<TypedMode>, String> {
        self.progress(handler)
    }
}

#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[derive(Debug, Clone, HirMode)]
pub struct TypedMode {
    dependencies: DepAna,
    types: Typed,
}

#[derive(Debug, Clone)]
pub struct Typed {
    stream_types: HashMap<SRef, StreamType>,
    expression_types: HashMap<ExprId, StreamType>,
    param_types: HashMap<(SRef, usize), ConcreteValueType>,
}

impl Typed {
    pub(crate) fn new(
        stream_types: HashMap<SRef, StreamType>,
        expression_types: HashMap<ExprId, StreamType>,
        param_types: HashMap<(SRef, usize), ConcreteValueType>,
    ) -> Self {
        Typed { stream_types, expression_types, param_types }
    }
}

#[mode_functionality]
pub trait TypedTrait {
    fn stream_type(&self, _sr: SRef) -> HirType;
    fn is_periodic(&self, _sr: SRef) -> bool;
    fn is_event(&self, _sr: SRef) -> bool;
    fn expr_type(&self, _eid: ExprId) -> HirType;
    /// Returns the Value Type of the `idx`-th Parameter for the Stream `stream`.
    fn get_parameter_type(&self, stream: SRef, idx: usize) -> ConcreteValueType;
}

impl HirStage for Hir<TypedMode> {
    type Error = ();
    type NextStage = OrderedMode;
    fn progress(self, _handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let order = Ordered::analyze(&self);

        let mode = OrderedMode { dependencies: self.mode.dependencies, types: self.mode.types, layers: order };

        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            expr_maps: self.expr_maps,
            mode,
        })
    }
}

impl Hir<TypedMode> {
    pub fn determine_evaluation_order(self, handler: &Handler) -> Result<Hir<OrderedMode>, ()> {
        self.progress(handler)
    }
}

#[derive(Debug, Clone)]
pub struct Ordered {
    event_layers: HashMap<SRef, StreamLayers>,
    periodic_layers: HashMap<SRef, StreamLayers>,
}
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[derive(Debug, Clone, HirMode)]
pub struct OrderedMode {
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
}
#[mode_functionality]
pub trait OrderedTrait {
    fn stream_layers(&self, sr: SRef) -> StreamLayers;
}

impl HirStage for Hir<OrderedMode> {
    type Error = ();
    type NextStage = MemBoundMode;
    fn progress(self, _handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let memory = MemBound::analyze(&self, false);

        let mode = MemBoundMode {
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: self.mode.layers,
            memory,
        };

        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            expr_maps: self.expr_maps,
            mode,
        })
    }
}

impl Hir<OrderedMode> {
    pub fn determine_memory_bounds(self, handler: &Handler) -> Result<Hir<MemBoundMode>, ()> {
        self.progress(handler)
    }
}

#[derive(Debug, Clone)]
pub struct MemBound {
    memory_bound_per_stream: HashMap<SRef, MemorizationBound>,
}

#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[covers_functionality(MemBoundTrait, memory)]
#[derive(Debug, Clone, HirMode)]
pub struct MemBoundMode {
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
    memory: MemBound,
}
#[mode_functionality]
pub trait MemBoundTrait {
    fn memory_bound(&self, sr: SRef) -> MemorizationBound;
}
impl HirStage for Hir<MemBoundMode> {
    type Error = ();
    type NextStage = CompleteMode;
    fn progress(self, _handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let mode = CompleteMode {
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: self.mode.layers,
            memory: self.mode.memory,
        };

        Ok(Hir {
            inputs: self.inputs,
            outputs: self.outputs,
            triggers: self.triggers,
            next_output_ref: self.next_output_ref,
            next_input_ref: self.next_input_ref,
            expr_maps: self.expr_maps,
            mode,
        })
    }
}

impl Hir<MemBoundMode> {
    pub fn finalize(self, handler: &Handler) -> Result<Hir<CompleteMode>, ()> {
        self.progress(handler)
    }
}
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[covers_functionality(MemBoundTrait, memory)]
#[derive(Debug, Clone, HirMode)]
pub struct CompleteMode {
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
    memory: MemBound,
}
