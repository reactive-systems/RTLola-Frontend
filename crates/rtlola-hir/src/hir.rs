/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occurs in different modes, adding different information to the intermediate representation.
*/
pub use crate::hir::expression::*;

pub mod expression;

pub mod naming;
mod print;

use crate::modes::HirMode;
use rtlola_reporting::Span;
use uom::si::rational64::Frequency as UOM_Frequency;

#[derive(Debug, Clone)]
pub struct RTLolaHir<M: HirMode> {
    pub(crate) inputs: Vec<Input>,
    pub(crate) outputs: Vec<Output>,
    pub(crate) triggers: Vec<Trigger>,
    pub(crate) next_input_ref: usize,
    pub(crate) next_output_ref: usize,
    pub(crate) mode: M,
}

pub type Hir<M> = RTLolaHir<M>;

impl<M: HirMode> Hir<M> {
    pub fn inputs(&self) -> impl Iterator<Item = &Input> {
        self.inputs.iter()
    }

    pub fn outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter()
    }

    pub fn triggers(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.iter()
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub fn num_triggers(&self) -> usize {
        self.triggers.len()
    }

    pub fn all_streams(&'_ self) -> impl Iterator<Item = SRef> + '_ {
        self.inputs
            .iter()
            .map(|i| i.sr)
            .chain(self.outputs.iter().map(|o| o.sr))
            .chain(self.triggers.iter().map(|t| t.sr))
    }
    pub fn get_input_with_name(&self, name: &str) -> Option<&Input> {
        self.inputs.iter().find(|&i| i.name == name)
    }
    pub fn get_output_with_name(&self, name: &str) -> Option<&Output> {
        self.outputs.iter().find(|&o| o.name == name)
    }
    pub fn output(&self, sref: SRef) -> Option<&Output> {
        self.outputs().find(|o| o.sr == sref)
    }
    pub fn input(&self, sref: SRef) -> Option<&Input> {
        self.inputs().find(|i| i.sr == sref)
    }
}

/**
An AST node representing the name of a called function and also the names of the arguments.
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionName {
    /**
    The name of the called function.
    */
    pub name: String,
    /**
    A list containing an element for each argument, containing the name if it is a named argument or else `None`.
    */
    pub arg_names: Vec<Option<String>>,
}

impl FunctionName {
    pub(crate) fn new(name: String, arg_names: &[Option<String>]) -> Self {
        Self { name, arg_names: Vec::from(arg_names) }
    }
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the stream.
    pub name: String,
    /// The reference pointing to this stream.
    pub sr: SRef,
    /// The user annotated Type
    pub annotated_type: AnnotatedType,
    /// The code span the input represents
    pub span: Span,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Output {
    /// The name of the stream.
    pub name: String,
    /// The user annotated Type
    pub annotated_type: Option<AnnotatedType>,
    /// The activation condition, which defines when a new value of a stream is computed. In periodic streams, the condition is 'None'
    pub activation_condition: Option<Ac>,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Parameter>,
    /// The declaration of the stream template for parametrized streams, e.g., the invoke declaration.
    pub instance_template: InstanceTemplate,
    /// The stream expression of a output stream, e.g., a + b.offset(by: -1).defaults(to: 0)
    pub expr_id: ExprId,
    /// The reference pointing to this stream.
    pub sr: SRef,
    /// The code span the output represents
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Parameter {
    /// The name of this parameter
    pub name: String,
    /// The annotated type of this parameter
    pub annotated_type: Option<AnnotatedType>,
    /// The id, index in the parameter vector in the output stream, for this parameter
    pub idx: usize,
    /// The code span of the parameter
    pub span: Span,
}

/// Use to hold either a frequency or an expression for the annotated activation condition
#[derive(Debug, Clone, PartialEq)]
pub enum Ac {
    Frequency { span: Span, value: UOM_Frequency },
    Expr(ExprId),
}

#[derive(Debug, Clone)]
pub struct InstanceTemplate {
    /// The invoke condition of the parametrized stream.
    pub spawn: Option<SpawnTemplate>,
    /// The extend condition of the parametrized stream.
    pub filter: Option<ExprId>,
    /// The termination condition of the parametrized stream.
    pub close: Option<ExprId>,
}

#[derive(Debug, Clone)]
pub struct SpawnTemplate {
    /// The expression defining the parameter instances. If the stream has more than one parameter, the expression needs to return a tuple, with one element for each parameter
    pub target: Option<ExprId>,
    /// The activation condition describing when a new instance is created.
    pub pacing: Option<Ac>,
    /// An additional condition for the creation of an instance, i.e., an instance is only created if the condition is true.
    pub condition: Option<ExprId>,
}

#[derive(Debug, Clone)]
pub struct Trigger {
    pub name: String,
    pub message: String,
    pub expr_id: ExprId,
    pub sr: SRef,
    /// The code span the trigger represents
    pub span: Span,
}

impl Trigger {
    pub(crate) fn new(
        name: Option<rtlola_parser::ast::Ident>,
        msg: Option<String>,
        expr_id: ExprId,
        sr: SRef,
        span: Span,
    ) -> Self {
        let name_str = name.map(|ident| ident.name).unwrap_or_else(String::new);
        Self { name: name_str, message: msg.unwrap_or_else(String::new), expr_id, sr, span }
    }
}

/// Represents the annotated given type for constants, input streams, etc.
/// It is converted from the AST type and an input for the typechecker.
/// After typechecking HirType is used to represent all type information.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AnnotatedType {
    //Can be annotated
    Int(u32),
    Float(u32),
    UInt(u32),
    Bool,
    String,
    Bytes,
    Option(Box<AnnotatedType>),
    Tuple(Vec<AnnotatedType>),
    //Used in function declaration
    Numeric,
    Param(usize, String),
}

/// Allows for referencing a window instance.
#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowReference {
    SlidingRef(usize),
    DiscreteRef(usize),
}

pub type WRef = WindowReference;

impl WindowReference {
    /// Provides access to the index inside the reference.
    pub fn idx(self) -> usize {
        match self {
            WindowReference::SlidingRef(u) => u,
            WindowReference::DiscreteRef(u) => u,
        }
    }
}

/// Allows for referencing an input stream within the specification.
pub type InputReference = usize;
/// Allows for referencing an output stream within the specification.
pub type OutputReference = usize;

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StreamReference {
    /// References an input stream.
    InRef(InputReference),
    /// References an output stream.
    OutRef(OutputReference),
}

pub type SRef = StreamReference;

impl StreamReference {
    /// Returns the index inside the reference if it is an output reference.  Panics otherwise.
    pub fn out_ix(&self) -> usize {
        match self {
            StreamReference::InRef(_) => unreachable!(),
            StreamReference::OutRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference if it is an input reference.  Panics otherwise.
    pub fn in_ix(&self) -> usize {
        match self {
            StreamReference::OutRef(_) => unreachable!(),
            StreamReference::InRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference disregarding whether it is an input or output reference.
    pub fn ix_unchecked(&self) -> usize {
        match self {
            StreamReference::InRef(ix) | StreamReference::OutRef(ix) => *ix,
        }
    }

    pub fn is_input(&self) -> bool {
        match self {
            StreamReference::OutRef(_) => false,
            StreamReference::InRef(_) => true,
        }
    }

    pub fn is_output(&self) -> bool {
        match self {
            StreamReference::OutRef(_) => true,
            StreamReference::InRef(_) => false,
        }
    }
}

impl PartialOrd for StreamReference {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (StreamReference::InRef(i), StreamReference::InRef(i2)) => Some(i.cmp(&i2)),
            (StreamReference::OutRef(o), StreamReference::OutRef(o2)) => Some(o.cmp(&o2)),
            (StreamReference::InRef(_), StreamReference::OutRef(_)) => Some(Ordering::Less),
            (StreamReference::OutRef(_), StreamReference::InRef(_)) => Some(Ordering::Greater),
        }
    }
}

impl Ord for StreamReference {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (StreamReference::InRef(i), StreamReference::InRef(i2)) => i.cmp(&i2),
            (StreamReference::OutRef(o), StreamReference::OutRef(o2)) => o.cmp(&o2),
            (StreamReference::InRef(_), StreamReference::OutRef(_)) => Ordering::Less,
            (StreamReference::OutRef(_), StreamReference::InRef(_)) => Ordering::Greater,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StreamAccessKind {
    Sync,
    DiscreteWindow(WRef),
    SlidingWindow(WRef),
    Hold,
    Offset(Offset),
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
use std::time::Duration;

use crate::modes::memory_bounds::MemorizationBound;
use crate::modes::ordering::Layer;

/// A trait for any kind of stream.
pub trait Stream {
    // Returns the spawn laying in which the stream is created.
    fn spawn_layer(&self) -> Layer;
    /// Returns the evaluation laying in which the stream resides.
    fn eval_layer(&self) -> Layer;
    /// Indicates whether or not the stream is an input stream.
    fn is_input(&self) -> bool;
    /// Indicates how many values need to be memorized.
    fn values_to_memorize(&self) -> MemorizationBound;
    /// Produces a stream references referring to the stream.
    fn as_stream_ref(&self) -> StreamReference;
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
