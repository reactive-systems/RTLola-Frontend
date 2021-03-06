pub(crate) mod ast_conversion;
pub(crate) mod dependencies;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod types;

use std::collections::HashMap;

use rtlola_reporting::Handler;

use self::dependencies::{DependencyErr, DependencyGraph, Streamdependencies, Windowdependencies};
use self::memory_bounds::MemBoundErr;
use self::ordering::OrderErr;
use self::types::HirType;
use crate::hir::{ExprId, Hir, SRef, WRef};
use crate::modes::memory_bounds::MemorizationBound;
use crate::modes::ordering::StreamLayers;
use crate::type_check::{ConcreteValueType, StreamType};

/// Defines the construct of a mode
///
/// This trait groups all available mode, adding different functionality to the [RtLolaHir](crate::RtLolaHir).
/// The trait [HirStage] declares the progress function that each mode needs to implement.
/// Each mode implements a separate trait defining the functionality that is added by the new mode, e.g., the [TypedMode] implements the [TypedTrait], providing an interface to get the types of a stream or expression.
/// With a new mode, a compiler flag derives the functionality of the previous modes.
/// The [RtLolaHir](crate::RtLolaHir) progesses the following modes:
/// [BaseMode] -> [DepAnaMode] -> [TypedMode] -> [OrderedMode] -> [MemBoundMode] -> [CompleteMode]
pub trait HirMode {}

/// Defines the functionality to progress one mode to the next one
pub trait HirStage: Sized {
    /// Defines the Error type of the `progress` function
    type Error;

    /// Defines the next mode that is produced by the `progress` function
    type NextStage: HirMode;

    /// Returns an [RtLolaHir](crate::RtLolaHir) with additional functionality
    fn progress(self, handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error>;
}

/// Represents the first stage in the [RtLolaHir](crate::RtLolaHir)(crate::RtLolaHir)
///
/// This struct represents the mode that is created with a new [RtLolaHir](crate::RtLolaHir)(crate::RtLolaHir).
/// The mode does not provide any additonal information and is therefore empty.
#[derive(Clone, Debug, HirMode, Copy)]
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
    /// Returns the [RtLolaHir](crate::RtLolaHir) with additional information about the dependencies between streams
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) after the dependency analysis.
    /// The new mode implements the same functionality as the [BaseMode] and additionally contains the dependencies between streams in the specification.
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    ///
    /// # Fails
    /// The function returns a [DependencyErr] if the specification is not well-formed.
    pub fn analyze_dependencies(self, handler: &Handler) -> Result<Hir<DepAnaMode>, DependencyErr> {
        self.progress(handler)
    }
}

/// Represents the results of the dependency analysis
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

/// Represents the mode after the dependency analysis
///
/// This struct represents the mode after the dependency analysis.
/// Besides this result, this mode has the same functionality as all the previous modes.
/// The [DepAnaTrait] defines the new functionality of the mode.
#[covers_functionality(DepAnaTrait, dependencies)]
#[derive(Debug, Clone, HirMode)]
pub struct DepAnaMode {
    dependencies: DepAna,
}

/// Describes the functionality of a mode after analyzing the dependencies
#[mode_functionality]
pub trait DepAnaTrait {
    /// Returns all streams that are direct accessed by `who`
    ///
    /// The function returns all streams that are direct accessed by `who`.
    /// A stream `who` accesses a stream `res`, if the stream expression, the spawn condition and definition, the filter condition, or the close condition of 'who' has a stream or window lookup to `res`.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accesses(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that are transitive accessed by `who`
    ///
    /// The function returns all streams that are transitive accessed by `who`.
    /// A stream `who` accesses a stream `res`, if the stream expression, the spawn condition and definition, the filter condition, or the close condition of 'who' has a stream or window lookup to 'res'.
    /// Transitive accesses are all accesses appearing in the expressions of the stream itself or indirect by another stream lookup.
    fn transitive_accesses(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that direct access `who`
    ///
    /// The function returns all streams that direct access `who`.
    /// A stream `who` is accessed by a stream `res`, if the stream expression, the spawn condition and definition, the filter condition, or the close condition of 'res' has a stream or window lookup to 'who'.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that transitive access `who`
    ///
    /// The function returns all streams that transitive access `who`.
    /// A stream `who` is accessed by a stream `res`, if the stream expression, the spawn condition and definition, the filter condition, or the close condition of 'res' has a stream or window lookup to 'who'.
    /// Transitive accesses are all accesses appearing in the expressions of the stream itself or indirect by another stream lookup.
    fn transitive_accessed_by(&self, who: SRef) -> Vec<SRef>;

    /// Returns all windows that aggregate `who` and the stream that uses the window
    ///
    /// The function returns all windows that aggregate `who` and the stream that uses the window.
    /// The result contains only the windows that are direct.
    fn aggregated_by(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    /// Returns all windows that are used in `who` and the corresponding stream that is aggregated
    ///
    /// The function returns all windows that are used in `who` and the corresponding stream that is aggregated.
    /// The result contains only the windows that are direct.
    fn aggregates(&self, who: SRef) -> Vec<(SRef, WRef)>; // (non-transitive)

    /// Returns the (Dependency Graph)[DependencyGraph] of the specification
    fn graph(&self) -> &DependencyGraph;
}

impl HirStage for Hir<DepAnaMode> {
    type Error = String;
    type NextStage = TypedMode;

    fn progress(self, handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let tts = crate::type_check::type_check(&self, handler)?;

        let mode = TypedMode {
            dependencies: self.mode.dependencies,
            types: tts,
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

impl Hir<DepAnaMode> {
    /// Returns the [RtLolaHir](crate::RtLolaHir) with the type information for each stream and expression
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) after the type analysis.
    /// The new mode implements the same functionality as the [DepAnaMode] and additionally holds for each stream and expression its [StreamType].
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    ///
    /// # Fails
    /// The function fails if the type checker finds a type error in the specification and returns a string with a detailed description.
    pub fn check_types(self, handler: &Handler) -> Result<Hir<TypedMode>, String> {
        self.progress(handler)
    }
}

/// Represents the results of the type checker
#[derive(Debug, Clone)]
pub struct Typed {
    stream_types: HashMap<SRef, StreamType>,
    expression_types: HashMap<ExprId, StreamType>,
    param_types: HashMap<(SRef, usize), ConcreteValueType>,
}

/// Represents the mode after the type checker call
///
/// This struct represents the mode after the type checker call.
/// Besides this result, this mode has the same functionality as all the previous modes.
/// The [TypedTrait] defines the new functionality of the mode.
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[derive(Debug, Clone, HirMode)]
pub struct TypedMode {
    dependencies: DepAna,
    types: Typed,
}

impl Typed {
    pub(crate) fn new(
        stream_types: HashMap<SRef, StreamType>,
        expression_types: HashMap<ExprId, StreamType>,
        param_types: HashMap<(SRef, usize), ConcreteValueType>,
    ) -> Self {
        Typed {
            stream_types,
            expression_types,
            param_types,
        }
    }
}

/// Describes the functionality of a mode after checking and inferring types
#[mode_functionality]
pub trait TypedTrait {
    /// Returns the [StreamType] of the given stream
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn stream_type(&self, sr: SRef) -> HirType;

    /// Returns true if the given stream has a periodic pacing type
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn is_periodic(&self, sr: SRef) -> bool;

    /// Returns true if the given stream has a event-based pacing type
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn is_event(&self, sr: SRef) -> bool;

    /// Returns the [StreamType] of the given expression
    ///
    /// # Panic
    /// The function panics if the [ExprId] is invalid.
    fn expr_type(&self, eid: ExprId) -> HirType;

    /// Returns the [ConcreteValueType] of the `idx` parameter of the `sr` stream template
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) or the index is invalid.
    fn get_parameter_type(&self, sr: SRef, idx: usize) -> ConcreteValueType;
}

impl HirStage for Hir<TypedMode> {
    type Error = OrderErr;
    type NextStage = OrderedMode;

    fn progress(self, _handler: &Handler) -> Result<Hir<Self::NextStage>, Self::Error> {
        let order = Ordered::analyze(&self);

        let mode = OrderedMode {
            dependencies: self.mode.dependencies,
            types: self.mode.types,
            layers: order,
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

impl Hir<TypedMode> {
    /// Returns the [RtLolaHir](crate::RtLolaHir) with the spawn and evaluation layer of each stream
    ///
    /// # Fails
    /// The function fails if the evaluation order cannot be determined.
    pub fn determine_evaluation_order(self, handler: &Handler) -> Result<Hir<OrderedMode>, OrderErr> {
        self.progress(handler)
    }
}

/// Represents the evaluation order
#[derive(Debug, Clone)]
pub struct Ordered {
    event_layers: HashMap<SRef, StreamLayers>,
    periodic_layers: HashMap<SRef, StreamLayers>,
}

/// Represents the mode after determining the evaluation order
///
/// This struct represents the mode after determining the evaluation order.
/// Besides this result, this mode has the same functionality as all the previous modes.
/// The [OrderedTrait] defines the new functionality of the mode.
#[covers_functionality(DepAnaTrait, dependencies)]
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(OrderedTrait, layers)]
#[derive(Debug, Clone, HirMode)]
pub struct OrderedMode {
    dependencies: DepAna,
    types: Typed,
    layers: Ordered,
}

/// Describes the functionality of a mode after computing the evaluation order
#[mode_functionality]
pub trait OrderedTrait {
    /// Returns the [StreamLayers] of the given stream
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn stream_layers(&self, sr: SRef) -> StreamLayers;
}

impl HirStage for Hir<OrderedMode> {
    type Error = MemBoundErr;
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
    /// Returns the [RtLolaHir](crate::RtLolaHir) with the memory-bound for each stream
    ///     
    /// # Fails
    /// The function fails if the memory cannot be determined.
    pub fn determine_memory_bounds(self, handler: &Handler) -> Result<Hir<MemBoundMode>, MemBoundErr> {
        self.progress(handler)
    }
}

/// Represents the results of the memory analysis
#[derive(Debug, Clone)]
pub struct MemBound {
    memory_bound_per_stream: HashMap<SRef, MemorizationBound>,
}

/// Represents the mode after the memory analysis
///
/// This struct represents the mode after the memory analysis.
/// Besides this result, this mode has the same functionality as all the previous modes.
/// The [MemBoundTrait] defines the new functionality of the mode.
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

/// Describes the functionality of a mode after computing the memory bounds
#[mode_functionality]
pub trait MemBoundTrait {
    /// Returns the memory bound of the given stream
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn memory_bound(&self, sr: SRef) -> MemorizationBound;
}

impl HirStage for Hir<MemBoundMode> {
    type Error = CompletionErr;
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

#[derive(Debug, Clone, Copy)]
pub enum CompletionErr {}

impl Hir<MemBoundMode> {
    /// Returns the [RtLolaHir](crate::RtLolaHir) in the last mode
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) in the [CompleteMode].
    /// This mode indicates that the [RtLolaHir](crate::RtLolaHir) has passed all analyzes and now contains all information.
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    pub fn finalize(self, handler: &Handler) -> Result<Hir<CompleteMode>, CompletionErr> {
        self.progress(handler)
    }
}

/// Represents the final mode.
///
/// This struct represents the final mode and indicates that the [RtLolaHir](crate::RtLolaHir) has passed all analyzes and now contains all information.
/// This mode has the same functionality as all the previous modes put together.
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
