pub(crate) mod ast_conversion;
pub(crate) mod dependencies;
pub(crate) mod memory_bounds;
pub(crate) mod ordering;
pub(crate) mod types;

use std::collections::HashMap;

use rtlola_reporting::RtLolaError;

use self::dependencies::{DependencyGraph, Origin, Streamdependencies, Transitivedependencies, Windowdependencies};
use self::types::HirType;
use crate::hir::{ExprId, Hir, SRef, StreamAccessKind, WRef};
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
/// [BaseMode] -> [TypedMode] -> [DepAnaMode] -> [OrderedMode] -> [MemBoundMode] -> [CompleteMode]
pub trait HirMode {}

/// Defines the functionality to progress one mode to the next one
pub trait HirStage: Sized {
    /// Defines the next mode that is produced by the `progress` function
    type NextStage: HirMode;

    /// Returns an [RtLolaHir](crate::RtLolaHir) with additional functionality
    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError>;
}

/// Represents the first stage in the [RtLolaHir](crate::RtLolaHir)(crate::RtLolaHir)
///
/// This struct represents the mode that is created with a new [RtLolaHir](crate::RtLolaHir)(crate::RtLolaHir).
/// The mode does not provide any additonal information and is therefore empty.
#[derive(Clone, Debug, HirMode, Copy)]
pub struct BaseMode {}

impl HirStage for Hir<BaseMode> {
    type NextStage = TypedMode;

    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError> {
        let tts = crate::type_check::type_check(&self)?;

        let mode = TypedMode { types: tts };

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
    /// Returns the [RtLolaHir](crate::RtLolaHir) with the type information for each stream and expression
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) after the type analysis.
    /// The new mode implements the same functionality as the [BaseMode] and additionally holds for each stream and expression its [StreamType].
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    ///
    /// # Fails
    /// The function fails if the type checker finds a type error in the specification and returns a string with a detailed description.
    pub fn check_types(self) -> Result<Hir<TypedMode>, RtLolaError> {
        self.progress()
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
#[covers_functionality(TypedTrait, types)]
#[derive(Debug, Clone, HirMode)]
pub struct TypedMode {
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

    /// Returns true if the given stream has a periodic evaluation pacing
    ///
    /// # Panic
    /// The function panics if the [StreamReference](crate::hir::StreamReference) is invalid.
    fn is_periodic(&self, sr: SRef) -> bool;

    /// Returns true if the given stream has a event-based evaluation pacing
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
    type NextStage = DepAnaMode;

    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError> {
        let dependencies = DepAna::analyze(&self)?;

        let mode = DepAnaMode {
            dependencies,
            types: self.mode.types,
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
    /// Returns the [RtLolaHir](crate::RtLolaHir) with additional information about the dependencies between streams
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) after the dependency analysis.
    /// The new mode implements the same functionality as the [TypedMode] and additionally contains the dependencies between streams in the specification.
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    ///
    /// # Fails
    /// The function returns a [RtLolaError] if the specification is not well-formed.
    pub fn analyze_dependencies(self) -> Result<Hir<DepAnaMode>, RtLolaError> {
        self.progress()
    }
}

/// Represents the results of the dependency analysis
#[derive(Debug, Clone)]
pub struct DepAna {
    direct_accesses: Streamdependencies,
    transitive_accesses: Transitivedependencies,
    direct_accessed_by: Streamdependencies,
    transitive_accessed_by: Transitivedependencies,
    aggregated_by: Windowdependencies,
    aggregates: Windowdependencies,
    graph: DependencyGraph,
}

/// Represents the mode after the dependency analysis
///
/// This struct represents the mode after the dependency analysis.
/// Besides this result, this mode has the same functionality as all the previous modes.
/// The [DepAnaTrait] defines the new functionality of the mode.
#[covers_functionality(TypedTrait, types)]
#[covers_functionality(DepAnaTrait, dependencies)]
#[derive(Debug, Clone, HirMode)]
pub struct DepAnaMode {
    types: Typed,
    dependencies: DepAna,
}

/// Describes the functionality of a mode after analyzing the dependencies
#[mode_functionality]
pub trait DepAnaTrait {
    /// Returns all streams that are direct accessed by `who`
    ///
    /// The function returns all streams that are direct accessed by `who`.
    /// A stream `who` accesses a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'who' has a stream or window lookup to `res`.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accesses(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that are direct accessed by `who` together with the corresponding stream access kinds.
    ///
    /// The function returns all streams that are direct accessed by `who` with all the stream access kinds that
    /// are used to access that stream.
    /// A stream `who` accesses a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'who' has a stream or window lookup to `res`.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accesses_with(&self, who: SRef) -> Vec<(SRef, Vec<(Origin, StreamAccessKind)>)>;

    /// Returns all streams that are transitive accessed by `who`
    ///
    /// The function returns all streams that are transitive accessed by `who`.
    /// A stream `who` accesses a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'who' has a stream or window lookup to 'res'.
    /// Transitive accesses are all accesses appearing in the expressions of the stream itself or indirect by another stream lookup.
    fn transitive_accesses(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that direct access `who`
    ///
    /// The function returns all streams that direct access `who`.
    /// A stream `who` is accessed by a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'res' has a stream or window lookup to 'who'.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accessed_by(&self, who: SRef) -> Vec<SRef>;

    /// Returns all streams that direct access `who` together with the corresponding stream access kinds.
    ///
    /// The function returns all streams that direct access `who` together with all the stream access kinds
    /// that they use to access `who`.
    /// A stream `who` is accessed by a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'res' has a stream or window lookup to 'who'.
    /// Direct accesses are all accesses appearing in the expressions of the stream itself.
    fn direct_accessed_by_with(&self, who: SRef) -> Vec<(SRef, Vec<(Origin, StreamAccessKind)>)>;

    /// Returns all streams that transitive access `who`
    ///
    /// The function returns all streams that transitive access `who`.
    /// A stream `who` is accessed by a stream `res`, if the stream expression, the spawn condition and definition, the evaluation condition, or the close condition of 'res' has a stream or window lookup to 'who'.
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
    type NextStage = OrderedMode;

    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError> {
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

impl Hir<DepAnaMode> {
    /// Returns the [RtLolaHir](crate::RtLolaHir) with the spawn and evaluation layer of each stream
    ///
    /// # Fails
    /// The function fails if the evaluation order cannot be determined.
    pub fn determine_evaluation_order(self) -> Result<Hir<OrderedMode>, RtLolaError> {
        self.progress()
    }
}

/// Represents the evaluation order
#[derive(Debug, Clone)]
pub struct Ordered {
    stream_layers: HashMap<SRef, StreamLayers>,
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
    type NextStage = MemBoundMode;

    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError> {
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
    pub fn determine_memory_bounds(self) -> Result<Hir<MemBoundMode>, RtLolaError> {
        self.progress()
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
    type NextStage = CompleteMode;

    fn progress(self) -> Result<Hir<Self::NextStage>, RtLolaError> {
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
    /// Returns the [RtLolaHir](crate::RtLolaHir) in the last mode
    ///
    /// The function returns the [RtLolaHir](crate::RtLolaHir) in the [CompleteMode].
    /// This mode indicates that the [RtLolaHir](crate::RtLolaHir) has passed all analyzes and now contains all information.
    /// The function moves the information of the previous mode to the new one and therefore destroys the current mode.
    pub fn finalize(self) -> Result<Hir<CompleteMode>, RtLolaError> {
        self.progress()
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
