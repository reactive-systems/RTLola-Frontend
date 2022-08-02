//! This module covers the High-Level Intermediate Representation (HIR) of an RTLola specification.
//!
//! The [RtLolaHir] is specifically designed to allow for convenient manipulation and analysis.  Hence, it is perfect for working *on* the specification rather than work *with* it.  
//! # Most Notable Structs and Enums
//! * [RtLolaMir](https://docs.rs/rtlola_frontend/struct.RtLolaMir.html) is the root data structure representing the specification.
//! * [Output] represents a single output stream.  The data structure is enriched with information regarding streams accessing it or accessed by it and much more.  For input streams confer [Input].
//! * [StreamReference] used for referencing streams within the Mir.
//! * [Expression] represents an expression.  It contains its [ExpressionKind] and its type.  The latter contains all information specific to a certain kind of expression such as sub-expressions of operators.
//!
//! # See Also
//! * [rtlola_frontend](https://docs.rs/rtlola_frontend) for an overview regarding different representations.
//! * [from_ast](crate::from_ast) / [fully_analyzed](crate::fully_analyzed) to obtain an [RtLolaHir] for a specification in form of an Ast.
//! * [RtLolaHir] for a data structs designed for working _on_it.
//! * [RtLolaAst](rtlola_parser::RtLolaAst), which is the most basic and down-to-syntax data structure available for RTLola.

mod expression;
mod print;
pub mod selector;

use std::collections::HashMap;
use std::time::Duration;

use rtlola_reporting::Span;
use serde::{Deserialize, Serialize};
use uom::si::rational64::Frequency as UOM_Frequency;

pub use crate::hir::expression::*;
pub use crate::modes::ast_conversion::TransformationErr;
pub use crate::modes::dependencies::{DependencyErr, DependencyGraph, EdgeWeight, Origin};
pub use crate::modes::memory_bounds::MemorizationBound;
pub use crate::modes::ordering::{Layer, StreamLayers};
use crate::modes::HirMode;
pub use crate::modes::{
    BaseMode, CompleteMode, DepAnaMode, DepAnaTrait, HirStage, MemBoundMode, MemBoundTrait, OrderedMode, OrderedTrait,
    TypedMode, TypedTrait,
};
use crate::stdlib::FuncDecl;
pub use crate::type_check::{
    ActivationCondition, ConcretePacingType, ConcreteStreamPacing, ConcreteValueType, StreamType,
};

/// This struct constitutes the Mid-Level Intermediate Representation (MIR) of an RTLola specification.
///
/// The [RtLolaHir] is specifically designed to allow for convenient manipulation and analysis.  Hence, it is perfect for working *on* the specification rather than work *with* it.  
///
/// # Most Notable Structs and Enums
/// * [RtLolaMir](https://docs.rs/rtlola_frontend/struct.RtLolaMir.html) is the root data structure representing the specification.
/// * [Output] represents a single output stream.  The data structure is enriched with information regarding streams accessing it or accessed by it and much more.  For input streams confer [Input].
/// * [StreamReference] used for referencing streams within the Mir.
/// * [Expression] represents an expression.  It contains its [ExpressionKind] and its type.  The latter contains all information specific to a certain kind of expression such as sub-expressions of operators.
///
/// # Type-State
/// The Hir follows a type-state pattern.  To this end, it has a type parameter, its HirMode.  The Hir starts in the [BaseMode] and progresses through different stages until reaching [CompleteMode].  
/// Each stage constitutes another level of refinement and adds functionality.  The functionality can be accesses by importing the respective trait and requiring the mode of the Hir to implement the trait.
/// The following traits exist.
/// * [DepAnaTrait] provides access to a dependency graph (see [petgraph](petgraph::stable_graph::StableGraph)) and functions to access immediate neighbors of streams. Obtained via [determine_evaluation_order](RtLolaHir::<TypeMode>::determine_evaluation_order).
/// * [TypedTrait] provides type information. Obtained via [check_types](crate::hir::RtLolaHir::<DepAnaMode>::check_types).
/// * [OrderedTrait] provides information regarding the evaluation order of streams. Obtained via [determine_evaluation_order](crate::hir::RtLolaHir::<TypedMode>::determine_evaluation_order).
/// * [MemBoundTrait] provides information on how many values of a stream have to be kept in memory at the same time. Obtained via [determine_memory_bounds](crate::hir::RtLolaHir::<OrderedMode>::determine_memory_bounds).
///
/// Progression through different stages is managed by the [HirStage] trait, in particular [HirStage::progress].
///
/// # See Also
/// * [rtlola_frontend](https://docs.rs/rtlola_frontend) for an overview regarding different representations.
/// * [from_ast](crate::from_ast) / [fully_analyzed](crate::fully_analyzed) to obtain an [RtLolaHir] for a specification in form of an Ast.
/// * [RtLolaHir] for a data structs designed for working _on_it.
/// * [RtLolaAst](rtlola_parser::RtLolaAst), which is the most basic and down-to-syntax data structure available for RTLola.
#[derive(Debug, Clone)]
pub struct RtLolaHir<M: HirMode> {
    /// Collection of input streams
    pub(crate) inputs: Vec<Input>,
    /// Collection of output streams
    pub(crate) outputs: Vec<Output>,
    /// Collection of trigger streams
    pub(crate) triggers: Vec<Trigger>,
    /// Next free input reference used to create new input streams
    pub(crate) next_input_ref: usize,
    /// Next free output reference used to create new output streams
    pub(crate) next_output_ref: usize,
    /// Maps expression ids to their expressions.
    pub(crate) expr_maps: ExpressionMaps,
    /// The current mode
    pub(crate) mode: M,
}

pub(crate) type Hir<M> = RtLolaHir<M>;

impl<M: HirMode> Hir<M> {
    /// Provides access to an iterator over all input streams.
    pub fn inputs(&self) -> impl Iterator<Item = &Input> {
        self.inputs.iter()
    }

    /// Provides access to an iterator over all output streams.
    pub fn outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter()
    }

    /// Provides access to an iterator over all triggers.
    pub fn triggers(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.iter()
    }

    /// Yields the number of input streams present in the Hir. Not necessarily equal to the number of input streams in the specification.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Yields the number of output streams present in the Hir.  Not necessarily equal to the number of output streams in the specification.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Yields the number of triggers present in the Hir.  Not necessarily equal to the number of triggers in the specification.
    pub fn num_triggers(&self) -> usize {
        self.triggers.len()
    }

    /// Provides access to an iterator over all streams, i.e., inputs, outputs, and triggers.
    pub fn all_streams(&'_ self) -> impl Iterator<Item = SRef> + '_ {
        self.inputs
            .iter()
            .map(|i| i.sr)
            .chain(self.outputs.iter().map(|o| o.sr))
            .chain(self.triggers.iter().map(|t| t.sr))
    }

    /// Retrieves an input stream based on its name.  Fails if no such input stream exists.
    pub fn get_input_with_name(&self, name: &str) -> Option<&Input> {
        self.inputs.iter().find(|&i| i.name == name)
    }

    /// Retrieves an output stream based on its name.  Fails if no such output stream exists.
    pub fn get_output_with_name(&self, name: &str) -> Option<&Output> {
        self.outputs.iter().find(|&o| o.name == name)
    }

    /// Retrieves an output stream based on a stream reference.  Fails if no such stream exists or `sref` is a [StreamReference::In].
    pub fn output(&self, sref: SRef) -> Option<&Output> {
        self.outputs().find(|o| o.sr == sref)
    }

    /// Retrieves an input stream based on a stream reference.  Fails if no such stream exists or `sref` is a [StreamReference::Out].
    pub fn input(&self, sref: SRef) -> Option<&Input> {
        self.inputs().find(|i| i.sr == sref)
    }

    /// Provides access to a collection of references for all windows occurring in the Hir.
    pub fn window_refs(&self) -> Vec<WRef> {
        self.expr_maps
            .sliding_windows
            .keys()
            .chain(self.expr_maps.discrete_windows.keys())
            .cloned()
            .collect()
    }

    /// Provides access to a collection of references for all sliding windows occurring in the Hir.
    pub fn sliding_windows(&self) -> Vec<&Window<SlidingAggr>> {
        self.expr_maps.sliding_windows.values().clone().collect()
    }

    /// Provides access to a collection of references for all discrete windows occurring in the Hir.
    pub fn discrete_windows(&self) -> Vec<&Window<DiscreteAggr>> {
        self.expr_maps.discrete_windows.values().clone().collect()
    }

    /// Retrieves an expression for a given expression id.
    ///
    /// # Panic
    /// Panics if the expression does not exist.
    pub fn expression(&self, id: ExprId) -> &Expression {
        &self.expr_maps.exprid_to_expr[&id]
    }

    /// Retrieves a function declaration for a given function name.
    ///
    /// # Panic
    /// Panics if the declaration does not exist.
    pub fn func_declaration(&self, func_name: &str) -> &FuncDecl {
        &self.expr_maps.func_table[func_name]
    }

    /// Retrieves a single sliding window for a given reference.  
    ///
    /// # Panic
    /// Panics if no such window exists.
    pub fn single_sliding(&self, window: WRef) -> Window<SlidingAggr> {
        *self
            .sliding_windows()
            .into_iter()
            .find(|w| w.reference == window)
            .unwrap()
    }

    /// Retrieves a single discrete window for a given reference.  
    ///
    /// # Panic
    /// Panics if no such window exists.
    pub fn single_discrete(&self, window: WRef) -> Window<DiscreteAggr> {
        *self
            .discrete_windows()
            .into_iter()
            .find(|w| w.reference == window)
            .unwrap()
    }

    /// Retrieves the spawn definition of a particular output stream or trigger or `None` for input references.
    pub fn spawn(&self, sr: SRef) -> Option<SpawnDef> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                let output = self.outputs.iter().find(|o| o.sr == sr);
                output.and_then(|o| o.spawn()).map(|st| {
                    SpawnDef::new(
                        st.expression.map(|e| self.expression(e)),
                        st.condition.map(|e| self.expression(e)),
                        st.pacing.as_ref(),
                    )
                })
            },
        }
    }

    /// Retrieves the spawn condition of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [SpawnDef] are needed, see [RtLolaHir::spawn]
    pub fn spawn_cond(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                self.outputs
                    .iter()
                    .find(|o| o.sr == sr)
                    .and_then(|o| o.spawn_cond())
                    .map(|eid| self.expression(eid))
            },
        }
    }

    /// Retrieves the spawn expresion of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [SpawnDef] are needed, see [RtLolaHir::spawn]
    pub fn spawn_expr(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                self.outputs
                    .iter()
                    .find(|o| o.sr == sr)
                    .and_then(|o| o.spawn_expr())
                    .map(|eid| self.expression(eid))
            },
        }
    }

    /// Retrieves the spawn pacing of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [SpawnDef] are needed, see [RtLolaHir::spawn]
    pub fn spawn_pacing(&self, sr: SRef) -> Option<&AnnotatedPacingType> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => self.outputs.iter().find(|o| o.sr == sr).and_then(|o| o.spawn_pacing()),
        }
    }

    /// Same behavior as [spawn].
    /// # Panic
    /// Panics if the stream does not exist or is an input/trigger.
    #[cfg(test)]
    pub(crate) fn spawn_unchecked(&self, sr: SRef) -> SpawnDef {
        self.spawn(sr).expect("Invalid for input and triggers references")
    }

    /// Retrieves the eval definition of a particular output stream or trigger or `None` for input references.
    pub fn eval(&self, sr: SRef) -> Option<EvalDef> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(idx) => {
                if idx < self.outputs.len() {
                    let output = self.outputs.iter().find(|o| o.sr == sr);
                    output.map(|o| {
                        let et = o.eval();
                        EvalDef::new(
                            et.condition.map(|id| self.expression(id)),
                            self.expression(et.expr),
                            et.annotated_pacing_type.as_ref(),
                        )
                    })
                } else {
                    // Trigger case
                    let tr = self.triggers().find(|tr| tr.sr == sr);
                    tr.map(|trigger| {
                        EvalDef::new(
                            None,
                            self.expression(trigger.expr_id),
                            trigger.annotated_pacing_type.as_ref(),
                        )
                    })
                }
            },
        }
    }

    /// Retrieves the eval condition of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [EvalDef] are needed, see [RtLolaHir::eval]
    pub fn eval_cond(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                self.outputs
                    .iter()
                    .find(|o| o.sr == sr)
                    .and_then(|o| o.eval_cond())
                    .map(|eid| self.expression(eid))
            },
        }
    }

    /// Retrieves the eval expression of a particular output stream or trigger and `None` for input references.
    /// If all parts of the [EvalDef] are needed, see [RtLolaHir::eval]
    pub fn eval_expr(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(o) => {
                if o < self.outputs.len() {
                    self.outputs
                        .iter()
                        .find(|o| o.sr == sr)
                        .map(|o| o.eval_expr())
                        .map(|eid| self.expression(eid))
                } else {
                    let tr = self.triggers().find(|tr| tr.sr == sr);
                    tr.map(|tr| tr.expr_id).map(|eid| self.expression(eid))
                }
            },
        }
    }

    /// Retrieves the eval pacing of a particular output stream or trigger `None` for input references.
    /// If all parts of the [EvalDef] are needed, see [RtLolaHir::eval]
    pub fn eval_pacing(&self, sr: SRef) -> Option<&AnnotatedPacingType> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(o) => {
                if o < self.outputs.len() {
                    self.outputs.iter().find(|o| o.sr == sr).and_then(|o| o.eval_pacing())
                } else {
                    let tr = self.triggers().find(|tr| tr.sr == sr);
                    tr.and_then(|tr| tr.annotated_pacing_type.as_ref())
                }
            },
        }
    }

    /// Same behavior as [`eval`](fn@Hir).
    /// # Panic
    /// Panics if the stream does not exist or is an input.
    pub(crate) fn eval_unchecked(&self, sr: StreamReference) -> EvalDef {
        self.eval(sr).expect("Invalid for input references")
    }

    /// Retrieves the expressions representing the close definition of a particular output stream or `None` for input and trigger references.
    pub fn close(&self, sr: SRef) -> Option<CloseDef> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                let ct = self.outputs.iter().find(|o| o.sr == sr).and_then(|o| o.close());
                ct.map(|ct| CloseDef::new(Some(self.expression(ct.condition)), ct.pacing.as_ref()))
            },
        }
    }

    /// Retrieves the expression representing the close condition of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [CloseDef] are needed, see [RtLolaHir::close]
    pub fn close_cond(&self, sr: SRef) -> Option<&Expression> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => {
                self.outputs
                    .iter()
                    .find(|o| o.sr == sr)
                    .and_then(|o| o.close_cond())
                    .map(|eid| self.expression(eid))
            },
        }
    }

    /// Retrieves the close pacing of a particular output stream or `None` for input and trigger references.
    /// If all parts of the [CloseDef] are needed, see [RtLolaHir::close]
    pub fn close_pacing(&self, sr: SRef) -> Option<&AnnotatedPacingType> {
        match sr {
            SRef::In(_) => None,
            SRef::Out(_) => self.outputs.iter().find(|o| o.sr == sr).and_then(|o| o.close_pacing()),
        }
    }

    /// Same behavior as [`close`](fn@Hir).
    /// # Panic
    /// Panics if the stream does not exist or is an input/trigger.
    #[cfg(test)]
    pub(crate) fn close_unchecked(&self, sr: StreamReference) -> CloseDef {
        self.close(sr).expect("Invalid for input and triggers references")
    }

    /// Generates a map from a [StreamReference] to the name of the corresponding stream.
    pub fn names(&self) -> HashMap<SRef, &str> {
        self.inputs()
            .map(|i| (i.sr, i.name.as_str()))
            .chain(self.outputs().map(|o| (o.sr, o.name.as_str())))
            .collect()
    }
}

/// A collection of maps for expression-related lookups, i.e., expressions, functions, and windows.
#[derive(Clone, Debug)]
pub(crate) struct ExpressionMaps {
    exprid_to_expr: HashMap<ExprId, Expression>,
    sliding_windows: HashMap<WRef, Window<SlidingAggr>>,
    discrete_windows: HashMap<WRef, Window<DiscreteAggr>>,
    func_table: HashMap<String, FuncDecl>,
}

impl ExpressionMaps {
    /// Creates a new expression map.
    pub(crate) fn new(
        exprid_to_expr: HashMap<ExprId, Expression>,
        sliding_windows: HashMap<WRef, Window<SlidingAggr>>,
        discrete_windows: HashMap<WRef, Window<DiscreteAggr>>,
        func_table: HashMap<String, FuncDecl>,
    ) -> Self {
        Self {
            exprid_to_expr,
            sliding_windows,
            discrete_windows,
            func_table,
        }
    }
}

/// Represents the name of a function including its arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionName {
    /// Name of the function
    pub name: String,
    /// The names of the arguments.  Each name might be empty.
    pub arg_names: Vec<Option<String>>,
}

impl FunctionName {
    /// Creates a new FunctionName.
    pub(crate) fn new(name: String, arg_names: &[Option<String>]) -> Self {
        Self {
            name,
            arg_names: Vec::from(arg_names),
        }
    }
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the stream.
    pub name: String,
    /// The reference pointing to this stream.
    pub(crate) sr: SRef,
    /// The user annotated Type
    pub(crate) annotated_type: AnnotatedType,
    /// The code span the input represents
    pub(crate) span: Span,
}

impl Input {
    /// Yields the reference referring to this input stream.
    pub fn sr(&self) -> StreamReference {
        self.sr
    }

    /// Yields the span referring to a part of the specification from which this stream originated.
    pub fn span(&self) -> Span {
        self.span.clone()
    }
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Output {
    /// The name of the stream.
    pub name: String,
    /// The user annotated Type
    pub(crate) annotated_type: Option<AnnotatedType>,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub(crate) params: Vec<Parameter>,
    /// The optional information on the spawning behavior of the stream
    pub(crate) spawn: Option<Spawn>,
    /// The information regarding evaluation expression and condition of the stream
    pub(crate) eval: Eval,
    /// The optional closing condition
    pub(crate) close: Option<Close>,
    /// The reference pointing to this stream.
    pub(crate) sr: SRef,
    /// The code span the output represents
    pub(crate) span: Span,
}

impl Output {
    /// Returns an iterator over the parameters of this stream.
    pub fn params(&self) -> impl Iterator<Item = &Parameter> {
        self.params.iter()
    }

    /// Yields the reference referring to this input stream.
    pub fn sr(&self) -> StreamReference {
        self.sr
    }

    /// Returns the [Spawn] template of the stream
    pub(crate) fn spawn(&self) -> Option<&Spawn> {
        self.spawn.as_ref()
    }

    /// Returns the expression id for the spawn condition of this stream
    /// If all parts of [Spawn] are required, see [spawn](fn@Hir)
    pub(crate) fn spawn_cond(&self) -> Option<ExprId> {
        self.spawn.as_ref().and_then(|st| st.condition)
    }

    /// Returns the expression id for the spawn expression of this stream
    /// If all parts of [Spawn] are required, see [spawn](fn@Hir)
    pub(crate) fn spawn_expr(&self) -> Option<ExprId> {
        self.spawn.as_ref().and_then(|st| st.expression)
    }

    /// Returns the pacing for the spawn condition of this stream
    /// If all parts of [Spawn] are required, see [spawn](fn@Hir)
    #[allow(dead_code)]
    pub(crate) fn spawn_pacing(&self) -> Option<&AnnotatedPacingType> {
        self.spawn.as_ref().and_then(|st| st.pacing.as_ref())
    }

    /// Returns the [Close] template of the stream
    pub(crate) fn close(&self) -> Option<&Close> {
        self.close.as_ref()
    }

    /// Returns the expression id for the close condition of this stream
    /// If all parts of [Close] are required, see [close](fn@Hir)
    pub(crate) fn close_cond(&self) -> Option<ExprId> {
        self.close.as_ref().map(|ct| ct.condition)
    }

    /// Returns the pacing for the close condition of this stream
    /// If all parts of [Close] are required, see [close](fn@Hir))
    #[allow(dead_code)]
    pub(crate) fn close_pacing(&self) -> Option<&AnnotatedPacingType> {
        self.close.as_ref().and_then(|ct| ct.pacing.as_ref())
    }

    /// Returns the [Eval] template of the stream
    pub(crate) fn eval(&self) -> &Eval {
        &self.eval
    }

    /// Returns the expression id for the evaluation condition of this stream
    pub(crate) fn eval_cond(&self) -> Option<ExprId> {
        self.eval.condition
    }

    /// Returns the expression id for the eval expression of this stream
    pub(crate) fn eval_expr(&self) -> ExprId {
        self.eval.expr
    }

    /// Returns the annotated pacing for the stream evaluation
    pub(crate) fn eval_pacing(&self) -> Option<&AnnotatedPacingType> {
        self.eval.annotated_pacing_type.as_ref()
    }

    /// Yields the span referring to a part of the specification from which this stream originated.
    pub fn span(&self) -> Span {
        self.span.clone()
    }
}

/// Represents a single parameter of a parametrized output stream.
#[derive(Debug, PartialEq, Clone, Eq)]
pub struct Parameter {
    /// The name of this parameter
    pub name: String,
    /// The annotated type of this parameter
    pub(crate) annotated_type: Option<AnnotatedType>,
    /// The index of this parameter
    pub(crate) idx: usize,
    /// The code span of the parameter
    pub(crate) span: Span,
}

impl Parameter {
    /// Yields the index of this parameter.  If the index is 3, then the parameter is the fourth parameter of the respective stream.
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Yields the span referring to a part of the specification where this parameter occurs.
    pub fn span(&self) -> Span {
        self.span.clone()
    }
}

/// Pacing information for stream; contains either a frequency or a condition on input streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotatedPacingType {
    /// The evaluation frequency
    Frequency {
        /// A span to the part of the specification containing the frequency
        span: Span,
        /// The actual frequency
        value: UOM_Frequency,
    },
    /// The expression which constitutes the condition under which the stream should be evaluated.
    Expr(ExprId),
}

/// Information regarding the spawning behavior of a stream
#[derive(Debug, Clone, Default)]
pub(crate) struct Spawn {
    /// The expression defining the parameter instances. If the stream has more than one parameter, the expression needs to return a tuple, with one element for each parameter
    pub(crate) expression: Option<ExprId>,
    /// The activation condition describing when a new instance is created.
    pub(crate) pacing: Option<AnnotatedPacingType>,
    /// An additional condition for the creation of an instance, i.e., an instance is only created if the condition is true.
    pub(crate) condition: Option<ExprId>,
}

impl Spawn {
    /// Returns a reference to the `Expression` representing the spawn expression if it exists
    pub(crate) fn spawn_expr<'a, M: HirMode>(&self, hir: &'a RtLolaHir<M>) -> Option<&'a Expression> {
        self.expression.map(|eid| hir.expression(eid))
    }

    /// Returns a vector of `Expression` references representing the expressions with which the parameters of the stream are initialized
    pub(crate) fn spawn_args<'a, M: HirMode>(&self, hir: &'a RtLolaHir<M>) -> Vec<&'a Expression> {
        self.spawn_expr(hir)
            .map(|se| {
                match &se.kind {
                    ExpressionKind::Tuple(spawns) => spawns.iter().collect(),
                    _ => vec![se],
                }
            })
            .unwrap_or_default()
    }

    /// Returns a reference to the `Expression` representing the spawn condition if it exists
    pub(crate) fn spawn_cond<'a, M: HirMode>(&self, hir: &'a RtLolaHir<M>) -> Option<&'a Expression> {
        self.condition.map(|eid| hir.expression(eid))
    }
}

/// Information regarding the evaluation condition and evaluation behavior of a stream
#[derive(Debug, Clone)]
pub(crate) struct Eval {
    /// The activation condition, which defines when a new value of a stream is computed.
    pub(crate) annotated_pacing_type: Option<AnnotatedPacingType>,
    /// The expression defining when an instance is evaluated
    pub(crate) condition: Option<ExprId>,
    /// The stream expression of a output stream, e.g., a + b.offset(by: -1).defaults(to: 0)
    pub(crate) expr: ExprId,
}

/// Information regarding the closing behavior of a stream
#[derive(Debug, Clone)]
pub(crate) struct Close {
    /// The expression defining if an instance is closed
    pub(crate) condition: ExprId,
    /// The activation condition describing when an instance is closed
    pub(crate) pacing: Option<AnnotatedPacingType>,
}

/// The Hir Spawn definition is composed of two optional expressions and the annotated pacing.
/// The first one refers to the spawn expression while the second one represents the spawn condition.
#[derive(Debug, Clone, Copy)]
pub struct SpawnDef<'a> {
    /// The expression of the stream is spawned with, setting the parameters, e.g. spawn with (3,x)
    pub expression: Option<&'a Expression>,
    /// The conditional expression of the spawn, e.g. when x > 5
    pub condition: Option<&'a Expression>,
    /// The pacing type  of the spawn, e.g. @1Hz or @input_i
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> SpawnDef<'a> {
    /// Constructs a new [SpawnDef]
    pub fn new(
        expression: Option<&'a Expression>,
        condition: Option<&'a Expression>,
        annotated_pacing: Option<&'a AnnotatedPacingType>,
    ) -> Self {
        Self {
            expression,
            condition,
            annotated_pacing,
        }
    }
}

/// The Hir Eval definition is composed of three expressions and the annotated pacing.
/// The first one refers to the evaluation condition, while the second one represents the evaluation expression, defining the value of the stream.
#[derive(Debug, Clone, Copy)]
pub struct EvalDef<'a> {
    /// The evaluation condition has to evaluated to true in order for the stream expression to be evaluated.
    pub condition: Option<&'a Expression>,
    /// The stream expression defines the computed value of the stream.
    pub expression: &'a Expression,
    /// The annotated pacing of the stream evaluation, describing when the condition and expression should be evaluated in a temporal manner.
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> EvalDef<'a> {
    /// Constructs a new [EvalDef]
    pub fn new(
        condition: Option<&'a Expression>,
        expr: &'a Expression,
        annotated_pacing: Option<&'a AnnotatedPacingType>,
    ) -> Self {
        Self {
            condition,
            expression: expr,
            annotated_pacing,
        }
    }
}

/// The Hir Close definition is composed of the Close condition expression and the annotated pacing.
#[derive(Debug, Clone, Copy)]
pub struct CloseDef<'a> {
    /// The close condition, defining when a stream instance is closed and no longer evaluated.
    pub condition: Option<&'a Expression>,
    /// The annotated pacing, indicating when the condition should be evaluated.
    pub annotated_pacing: Option<&'a AnnotatedPacingType>,
}

impl<'a> CloseDef<'a> {
    /// Constructs a new [CloseDef]
    pub fn new(condition: Option<&'a Expression>, annotated_pacing: Option<&'a AnnotatedPacingType>) -> Self {
        Self {
            condition,
            annotated_pacing,
        }
    }
}

/// Represents a trigger of an RTLola specification.
#[derive(Debug, Clone)]
pub struct Trigger {
    /// The message that will be conveyed when the trigger expression evaluates to true.
    pub message: String,
    /// A collection of streams which can be used in the message. Their value is printed when the trigger is activated.
    pub info_streams: Vec<StreamReference>,
    /// The activation condition, which defines when the trigger is evaluated.
    pub(crate) annotated_pacing_type: Option<AnnotatedPacingType>,
    /// The id of the expression belonging to the trigger
    pub(crate) expr_id: ExprId,
    /// A reference to the stream which represents this trigger.
    pub(crate) sr: SRef,
    /// The code span the trigger represents
    pub(crate) span: Span,
}

impl Trigger {
    /// Creates a new trigger.
    pub(crate) fn new(
        msg: Option<String>,
        infos: Vec<StreamReference>,
        pt: Option<AnnotatedPacingType>,
        expr_id: ExprId,
        sr: SRef,
        span: Span,
    ) -> Self {
        Self {
            info_streams: infos,
            annotated_pacing_type: pt,
            message: msg.unwrap_or_default(),
            expr_id,
            sr,
            span,
        }
    }

    /// Provides the reference of a stream that represents this trigger.
    pub fn sr(&self) -> StreamReference {
        self.sr
    }

    /// Provides access to the trigger condition
    pub fn expression(&self) -> ExprId {
        self.expr_id
    }

    /// The code span referring to the original location of the trigger in the specification.
    pub fn span(&self) -> Span {
        self.span.clone()
    }
}

/// Represents the annotated given type for constants, input streams, etc.
/// It is converted from the AST type and an input for the type checker.
/// After typechecking HirType is used to represent all type information.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub(crate) enum AnnotatedType {
    Int(u32),
    Float(u32),
    UInt(u32),
    Bool,
    String,
    Bytes,
    Option(Box<AnnotatedType>),
    Tuple(Vec<AnnotatedType>),
    Numeric,
    Sequence,
    Param(usize, String),
}

impl AnnotatedType {
    /// Yields a collection of primitive types and their names.
    pub(crate) fn primitive_types() -> Vec<(&'static str, &'static AnnotatedType)> {
        let mut types = vec![];
        types.extend_from_slice(&crate::stdlib::PRIMITIVE_TYPES);
        types.extend_from_slice(&crate::stdlib::PRIMITIVE_TYPES_ALIASES);

        types
    }
}

/// Allows for referencing a window instance.
#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowReference {
    /// Refers to a sliding window
    Sliding(usize),
    /// Refers to a discrete window
    Discrete(usize),
}

pub(crate) type WRef = WindowReference;

impl WindowReference {
    /// Provides access to the index inside the reference.
    pub fn idx(self) -> usize {
        match self {
            WindowReference::Sliding(u) => u,
            WindowReference::Discrete(u) => u,
        }
    }
}

/// Allows for referencing an input stream within the specification.
pub type InputReference = usize;
/// Allows for referencing an output stream within the specification.
pub type OutputReference = usize;

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamReference {
    /// References an input stream.
    In(InputReference),
    /// References an output stream.
    Out(OutputReference),
}

pub(crate) type SRef = StreamReference;

impl StreamReference {
    /// Returns the index inside the reference if it is an output reference.  Panics otherwise.
    pub fn out_ix(&self) -> usize {
        match self {
            StreamReference::In(_) => unreachable!(),
            StreamReference::Out(ix) => *ix,
        }
    }

    /// Returns the index inside the reference if it is an input reference.  Panics otherwise.
    pub fn in_ix(&self) -> usize {
        match self {
            StreamReference::Out(_) => unreachable!(),
            StreamReference::In(ix) => *ix,
        }
    }

    /// Returns the index inside the reference disregarding whether it is an input or output reference.
    pub fn ix_unchecked(&self) -> usize {
        match self {
            StreamReference::In(ix) | StreamReference::Out(ix) => *ix,
        }
    }

    /// True if the reference is an instance of [StreamReference::In], false otherwise.
    pub fn is_input(&self) -> bool {
        match self {
            StreamReference::Out(_) => false,
            StreamReference::In(_) => true,
        }
    }

    /// True if the reference is an instance of [StreamReference::Out], false otherwise.
    pub fn is_output(&self) -> bool {
        match self {
            StreamReference::Out(_) => true,
            StreamReference::In(_) => false,
        }
    }
}

impl PartialOrd for StreamReference {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (StreamReference::In(i), StreamReference::In(i2)) => Some(i.cmp(i2)),
            (StreamReference::Out(o), StreamReference::Out(o2)) => Some(o.cmp(o2)),
            (StreamReference::In(_), StreamReference::Out(_)) => Some(Ordering::Less),
            (StreamReference::Out(_), StreamReference::In(_)) => Some(Ordering::Greater),
        }
    }
}

impl Ord for StreamReference {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (StreamReference::In(i), StreamReference::In(i2)) => i.cmp(i2),
            (StreamReference::Out(o), StreamReference::Out(o2)) => o.cmp(o2),
            (StreamReference::In(_), StreamReference::Out(_)) => Ordering::Less,
            (StreamReference::Out(_), StreamReference::In(_)) => Ordering::Greater,
        }
    }
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Offset {
    /// A strictly positive discrete offset, e.g., `4`, or `42`
    FutureDiscrete(u32),
    /// A non-negative discrete offset, e.g., `0`, `-4`, or `-42`
    PastDiscrete(u32),
    /// A positive real-time offset, e.g., `-3ms`, `-4min`, `-2.3h`
    FutureRealTime(Duration),
    /// A non-negative real-time offset, e.g., `0`, `4min`, `2.3h`
    PastRealTime(Duration),
}

impl Offset {
    /// Returns `true`, iff the Offset is negative
    pub(crate) fn has_negative_offset(&self) -> bool {
        match self {
            Offset::FutureDiscrete(_) | Offset::FutureRealTime(_) => false,
            Offset::PastDiscrete(o) => *o != 0,
            Offset::PastRealTime(o) => o.as_nanos() != 0,
        }
    }

    pub(crate) fn as_memory_bound(&self, dynamic: bool) -> MemorizationBound {
        match self {
            Offset::PastDiscrete(o) => MemorizationBound::Bounded(*o) + MemorizationBound::default_value(dynamic),
            Offset::FutureDiscrete(_) => unimplemented!(),
            Offset::FutureRealTime(_) => unimplemented!(),
            Offset::PastRealTime(_) => unimplemented!(),
        }
    }
}

impl PartialOrd for Offset {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        use Offset::*;
        match (self, other) {
            (PastDiscrete(_), FutureDiscrete(_))
            | (PastRealTime(_), FutureRealTime(_))
            | (PastDiscrete(_), FutureRealTime(_))
            | (PastRealTime(_), FutureDiscrete(_)) => Some(Ordering::Less),

            (FutureDiscrete(_), PastDiscrete(_))
            | (FutureDiscrete(_), PastRealTime(_))
            | (FutureRealTime(_), PastDiscrete(_))
            | (FutureRealTime(_), PastRealTime(_)) => Some(Ordering::Greater),

            (FutureDiscrete(a), FutureDiscrete(b)) => Some(a.cmp(b)),
            (PastDiscrete(a), PastDiscrete(b)) => Some(b.cmp(a)),

            (_, _) => unimplemented!(),
        }
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
