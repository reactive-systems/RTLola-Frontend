//! This module covers the Mid-Level Intermediate Representation (MIR) of an RTLola specification.
//!
//! The [RtLolaMir] is specifically designed to allow convenient navigation and access to data.  Hence, it is perfect for working *with* the specification
//! rather than work *on* it.  
//!
//! # Most Notable Structs and Enums
//! * [RtLolaMir] is the root data structure representing the specification.
//! * [OutputStream] represents a single output stream.  The data structure is enriched with information regarding streams accessing it or accessed by it and much more.  For input streams confer [InputStream].
//! * [StreamReference] used for referencing streams within the Mir.
//! * [Spawn] and [Close] contain all information regarding the parametrization, spawning and closing behavior of streams.
//! * [Eval] contains the information regarding the evaluation condition and the expression of the stream.
//! * [Expression] represents an expression.  It contains its [ExpressionKind] and its type.  The latter contains all information specific to a certain kind of expression such as sub-expressions of operators.
//!
//! # See Also
//! * [rtlola_frontend](crate) for an overview regarding different representations.
//! * [rtlola_frontend::parse](crate::parse) to obtain an [RtLolaMir] for a specification in form of a string or path to a specification file.
//! * [rtlola_hir::hir::RtLolaHir] for a data structs designed for working _on_it.
//! * [RtLolaAst](rtlola_parser::RtLolaAst), which is the most basic and down-to-syntax data structure available for RTLola.

mod dependency_graph;
mod print;
mod schedule;

use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::{Deref, DerefMut};
use std::time::Duration;

use itertools::Either;
use num::traits::Inv;
pub use print::RtLolaMirPrinter;
use rtlola_hir::hir::ConcreteValueType;
pub use rtlola_hir::hir::{
    Layer, MemBoundMode, MemorizationBound, Origin, OutputKind, RtLolaHir, StreamLayers,
    WindowReference,
};
pub use rtlola_parser::ast::Tag;
#[cfg(feature = "spanned")]
use rtlola_reporting::Span;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uom::si::rational64::{Frequency as UOM_Frequency, Time as UOM_Time};
use uom::si::time::nanosecond;

pub use self::dependency_graph::DependencyGraph;
pub use crate::mir::schedule::{Deadline, Schedule, Task};

pub(crate) type Mir = RtLolaMir;

/// A trait for any kind of stream.
pub trait Stream {
    /// Reports the evaluation layer of the spawn condition of the stream.
    fn spawn_layer(&self) -> Layer;
    /// Reports the evaluation layer of the stream.
    fn eval_layer(&self) -> Layer;
    /// Reports the name of the stream.
    fn name(&self) -> &str;
    /// Returns the type of the stream.
    fn ty(&self) -> &Type;
    /// Indicates whether or not the stream is an input stream.
    fn is_input(&self) -> bool;
    /// Indicates whether or not the stream has parameters.
    fn is_parameterized(&self) -> bool;
    /// Indicates whether or not the stream spawned / dynamically created.
    fn is_spawned(&self) -> bool;
    /// Indicates whether or not the stream is closed.
    fn is_closed(&self) -> bool;
    /// Indicated whether or not the stream is filtered.
    fn is_eval_filtered(&self) -> bool;
    /// Indicates how many values of the stream's [Type] need to be memorized.
    fn values_to_memorize(&self) -> MemorizationBound;
    /// Produces a stream references referring to the stream.
    fn as_stream_ref(&self) -> StreamReference;
    /// Returns the collection of streams that access the stream non-transitively.
    fn accessed_by(&self) -> &Accesses;
    /// Returns the collection of sliding windows that access the stream non-transitively.
    /// This includes both sliding and discrete windows.
    fn aggregated_by(&self) -> &[(StreamReference, Origin, WindowReference)];
    /// Returns the collection of sliding windows that are accessed by the stream non-transitively.
    /// This includes both sliding and discrete windows.
    fn aggregates(&self) -> &[(StreamReference, Origin, WindowReference)];
    /// Returns the tags annotated to this stream.
    fn tags(&self) -> &HashMap<String, Option<String>>;
    #[cfg(feature = "spanned")]
    /// Returns the spans of all tags annotated to this stream.
    fn tags_span(&self) -> &HashMap<String, Span>;
}

/// This struct constitutes the Mid-Level Intermediate Representation (MIR) of an RTLola specification.
///
/// The [RtLolaMir] is specifically designed to allow convenient navigation and access to data.  Hence, it is perfect for working _with_ the specification
/// rather than work _on_ it.  
///
/// # Most Notable Structs and Enums
/// * [Stream] is a trait offering several convenient access methods for everything constituting a stream.
/// * [OutputStream] represents a single output stream.  The data structure is enriched with information regarding streams accessing it or accessed by it and much more.  For input streams confer [InputStream].
/// * [StreamReference] used for referencing streams within the Mir.
/// * [Spawn] and [Close] contain all information regarding the parametrization, spawning and closing behavior of streams.
/// * [Eval] contains the information regarding the evaluation condition and the expression of the stream. The [Expression] represents an computational evaluation.  It contains its [ExpressionKind] and its type.  The latter contains all information specific to a certain kind of expression such as sub-expressions of operators.
///
/// # See Also
/// * [rtlola_frontend](crate) for an overview regarding different representations.
/// * [rtlola_frontend::parse](crate::parse) to obtain an [RtLolaMir] for a specification in form of a string or path to a specification file.
/// * [rtlola_hir::hir::RtLolaHir] for a data structs designed for working _on_it.
/// * [RtLolaAst](rtlola_parser::RtLolaAst), which is the most basic and down-to-syntax data structure available for RTLola.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RtLolaMir {
    /// Contains all input streams.
    pub inputs: Vec<InputStream>,
    /// Contains all unparameterized output streams including all triggers.  They only contain the information relevant for every single kind of output stream.  Refer to [RtLolaMir::time_driven], [RtLolaMir::event_driven],
    /// and [RtLolaMir::triggers] for more information.
    pub unparameterized_outputs: Vec<UnparameterizedOutputStream>,
    /// Contains all parameterized output streams including all triggers.  They only contain the information relevant for every single kind of output stream.  Refer to [RtLolaMir::time_driven], [RtLolaMir::event_driven],
    /// and [RtLolaMir::triggers] for more information.
    pub parameterized_outputs: Vec<ParameterizedOutputStream>,
    /// References and pacing information of all time-driven streams.
    pub time_driven: Vec<TimeDrivenStream>,
    /// References and pacing information of all event-driven streams.
    pub event_driven: Vec<EventDrivenStream>,
    /// A collection of all discrete windows.
    pub discrete_windows: Vec<DiscreteWindow>,
    /// A collection of all sliding windows.
    pub sliding_windows: Vec<SlidingWindow>,
    /// A collection of all instance aggregations.
    pub instance_aggregations: Vec<InstanceAggregation>,
    /// The references of all outputs that represent triggers
    pub triggers: Vec<Trigger>,
    /// The global tags of the specification
    pub global_tags: Tags,
    #[cfg(feature = "spanned")]
    /// The span's of the global tags
    pub global_tags_span: HashMap<String, Span>,
}

/// Represents an RTLola value type.  This does not including pacing information, for this refer to [TimeDrivenStream] and [EventDrivenStream].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum Type {
    /// A boolean type
    Bool,
    /// An integer type of fixed bit-width
    Int(IntTy),
    /// An unsigned integer type of fixed bit-width
    UInt(UIntTy),
    /// A floating point type of fixed bit-width
    Float(FloatTy),
    /// A signed fixed point type of fixed bit-width
    Fixed(FixedTy),
    /// An unsigned fixed point type of fixed bit-width
    UFixed(FixedTy),
    /// A unicode string
    String,
    /// A sequence of 8-bit bytes
    Bytes,
    /// An n-ary tuples where n is the length of the contained vector
    Tuple(Vec<Type>),
    /// An optional value type, e.g., resulting from accessing a past value of a stream
    Option(Box<Type>),
    /// A type describing a function
    Function {
        /// The types of the arguments to the function, monomorphized
        args: Vec<Type>,
        /// The monomorphized return type of the function
        ret: Box<Type>,
    },
    /// A 2-element vector
    Vec2,
    /// A 3-element vector
    Vec3,
}

/// Represents an RTLola pacing type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacingType {
    /// Represents a periodic pacing with a fixed global frequency
    GlobalPeriodic(UOM_Frequency),
    /// Represents a periodic pacing with a fixed local frequency
    LocalPeriodic(UOM_Frequency),
    /// Represents an event based pacing defined by an [ActivationCondition]
    Event(ActivationCondition),
    /// The pacing is constant, meaning that the value is always present.
    Constant,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum IntTy {
    /// Represents an 8-bit integer.
    Int8,
    /// Represents a 16-bit integer.
    Int16,
    /// Represents a 32-bit integer.
    Int32,
    /// Represents a 64-bit integer.
    Int64,
    /// Represents a 128-bit integer.
    Int128,
    /// Represents a 256-bit integer.
    Int256,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum UIntTy {
    /// Represents an 8-bit unsigned integer.
    UInt8,
    /// Represents a 16-bit unsigned integer.
    UInt16,
    /// Represents a 32-bit unsigned integer.
    UInt32,
    /// Represents a 64-bit unsigned integer.
    UInt64,
    /// Represents a 128-bit unsigned integer.
    UInt128,
    /// Represents a 256-bit unsigned integer.
    UInt256,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum FloatTy {
    /// Represents a 32-bit floating point number.
    Float32,
    /// Represents a 64-bit floating point number.
    Float64,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum FixedTy {
    /// Represents a 64-bit fixed point number with 32 integer bits and 32 fractional bits
    Fixed64_32,
    /// Represents a 32-bit fixed point number with 16 integer bits and 16 fractional bits
    Fixed32_16,
    /// Represents a 16-bit fixed point number with 16 integer bits and 8 fractional bits
    Fixed16_8,
}

impl From<ConcreteValueType> for Type {
    fn from(ty: ConcreteValueType) -> Type {
        match ty {
            ConcreteValueType::Integer8 => Type::Int(IntTy::Int8),
            ConcreteValueType::Integer16 => Type::Int(IntTy::Int16),
            ConcreteValueType::Integer32 => Type::Int(IntTy::Int32),
            ConcreteValueType::Integer64 => Type::Int(IntTy::Int64),
            ConcreteValueType::UInteger8 => Type::UInt(UIntTy::UInt8),
            ConcreteValueType::UInteger16 => Type::UInt(UIntTy::UInt16),
            ConcreteValueType::UInteger32 => Type::UInt(UIntTy::UInt32),
            ConcreteValueType::UInteger64 => Type::UInt(UIntTy::UInt64),
            ConcreteValueType::Float32 => Type::Float(FloatTy::Float32),
            ConcreteValueType::Float64 => Type::Float(FloatTy::Float64),
            ConcreteValueType::Tuple(t) => Type::Tuple(t.into_iter().map(Type::from).collect()),
            ConcreteValueType::TString => Type::String,
            ConcreteValueType::Byte => Type::Bytes,
            ConcreteValueType::Option(o) => Type::Option(Box::new(Type::from(*o))),
            _ => unreachable!("cannot lower `ValueTy` {}", ty),
        }
    }
}

type Accesses = Vec<(StreamReference, Vec<(Origin, StreamAccessKind)>)>;

/// Allows for referencing a stream within the specification.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Copy, Hash, PartialOrd, Ord)]
pub enum StreamReference {
    /// References an input stream.
    In(InputReference),
    /// References an output stream.
    Out(OutputReference),
}

impl StreamReference {
    /// Returns the index inside the reference if it is an output reference.  Panics otherwise.
    pub fn out_ix(&self) -> OutputReference {
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

/// Allows for referencing an input stream within the specification.
pub type InputReference = usize;

/// Allows for referencing an output stream within the specification.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Copy, Hash, PartialOrd, Ord)]
pub enum OutputReference {
    /// The output stream is unparameterized
    Unparameterized(usize),
    /// The output stream is parameterized
    Parameterized(usize),
}

impl OutputReference {
    /// Returns the StreamReference of the output
    pub fn sr(&self) -> StreamReference {
        StreamReference::Out(*self)
    }

    /// Returns the index for an unparameterized stream
    pub fn unparameterized_idx(&self) -> usize {
        match &self {
            OutputReference::Unparameterized(i) => *i,
            OutputReference::Parameterized(_) => panic!(),
        }
    }

    /// Returns the index for an parameterized stream
    pub fn parameterized_idx(&self) -> usize {
        match &self {
            OutputReference::Parameterized(i) => *i,
            OutputReference::Unparameterized(_) => panic!(),
        }
    }
}

/// Contains all information inherent to an input stream.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct InputStream {
    /// The name of the stream
    pub name: String,
    /// The value type of the stream.  Note that its pacing is always pre-determined.
    pub ty: Type,
    /// The collection of streams that access the current stream non-transitively
    pub accessed_by: Accesses,
    /// The collection of sliding windows that access this stream non-transitively.  This includes both sliding and discrete windows.
    pub aggregated_by: Vec<(StreamReference, Origin, WindowReference)>,
    /// The collection of windows that is accessed by this stream.  This includes both sliding and discrete windows.
    pub aggregates: Vec<(StreamReference, Origin, WindowReference)>,
    /// Provides the evaluation of layer of this stream.
    pub layer: StreamLayers,
    /// Provides the number of values of this stream's type that need to be memorized.  Refer to [Type::size] to get a type's byte-size.
    pub memory_bound: MemorizationBound,
    /// The reference referring to this stream
    pub reference: InputReference,
    /// The tags annotated to this stream.
    pub tags: Tags,
    #[cfg(feature = "spanned")]
    /// The span of the tags annotated to the input stream
    pub tags_span: HashMap<String, Span>,
    #[cfg(feature = "spanned")]
    /// The span of the input stream definition
    pub span: Span,
}

/// Contains all information relevant to every kind of output stream.
///
/// Refer to [TimeDrivenStream], [EventDrivenStream], and [Trigger], as well as their respective fields in the Mir for additional information.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct CommonOutputStream {
    /// The name of the stream.
    pub name: String,
    /// The kind of the output (regular output or trigger)
    pub kind: OutputKind,
    /// The value type of the stream.
    pub ty: Type,
    /// Information on the spawn behavior of the stream
    pub spawn: Spawn,
    /// Information on the evaluation behavior of the stream
    pub eval: Eval,
    /// The condition under which the stream is supposed to be closed
    pub close: Close,
    /// The collection of streams this stream accesses non-transitively.  Includes this stream's spawn, evaluation condition, and close expressions.
    pub accesses: Accesses,
    /// The collection of streams that access the current stream non-transitively
    pub accessed_by: Accesses,
    /// The collection of windows that access this stream non-transitively.  This includes both sliding and discrete windows.
    pub aggregated_by: Vec<(StreamReference, Origin, WindowReference)>,
    /// The collection of windows that is accessed by this stream.  This includes both sliding and discrete windows.
    pub aggregates: Vec<(StreamReference, Origin, WindowReference)>,
    /// Provides the number of values of this stream's type that need to be memorized.  Refer to [Type::size] to get a type's byte-size.
    pub memory_bound: MemorizationBound,
    /// Provides the evaluation of layer of this stream.
    pub layer: StreamLayers,
    /// The reference referring to this stream
    pub reference: OutputReference,
    /// The tags annotated to this stream.
    pub tags: Tags,
    #[cfg(feature = "spanned")]
    /// The span of the tags annotated to the output stream
    pub tags_span: HashMap<String, Span>,
    #[cfg(feature = "spanned")]
    /// The span of the output stream definition
    pub span: Span,
}

/// Contains all information related to parameterized output streams
///
/// Refer to [OutputStream] as well as the respective fields in the Mir for additional information.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ParameterizedOutputStream {
    /// The expression needs to be evaluated whenever the stream with this Spawn template is supposed to be spawned.  The result of the evaluation constitutes the respective parameters.
    pub spawn_expr: Expression,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Parameter>,
    /// The underlying output stream information
    pub output: CommonOutputStream,
}

impl Deref for ParameterizedOutputStream {
    type Target = CommonOutputStream;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}

impl DerefMut for ParameterizedOutputStream {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.output
    }
}

/// Represents an unparameterized output stream
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnparameterizedOutputStream(pub CommonOutputStream);

impl Deref for UnparameterizedOutputStream {
    type Target = CommonOutputStream;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnparameterizedOutputStream {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Trait to collect all common functionality of output streams
pub trait OutputStream: Deref<Target = CommonOutputStream> + Stream {
    /// Turns the output stream into Either<Unparameterized, ParameterizedOutputStream>
    fn into_either(self) -> Either<UnparameterizedOutputStream, ParameterizedOutputStream>;
}

impl OutputStream for UnparameterizedOutputStream {
    fn into_either(self) -> Either<UnparameterizedOutputStream, ParameterizedOutputStream> {
        Either::Left(self)
    }
}

impl OutputStream for ParameterizedOutputStream {
    fn into_either(self) -> Either<UnparameterizedOutputStream, ParameterizedOutputStream> {
        Either::Right(self)
    }
}

impl OutputStream for Either<UnparameterizedOutputStream, ParameterizedOutputStream> {
    fn into_either(self) -> Either<UnparameterizedOutputStream, ParameterizedOutputStream> {
        self
    }
}

impl OutputStream for Either<ParameterizedOutputStream, UnparameterizedOutputStream> {
    fn into_either(self) -> Either<UnparameterizedOutputStream, ParameterizedOutputStream> {
        match self {
            Either::Left(lhs) => Either::Right(lhs),
            Either::Right(rhs) => Either::Left(rhs),
        }
    }
}

trait EitherStream {
    fn as_stream(&self) -> &dyn OutputStream;
}

macro_rules! implement_either {
    ($lhs:ty, $rhs:ty) => {
        impl EitherStream for Either<$lhs, $rhs> {
            fn as_stream(&self) -> &dyn OutputStream {
                match self {
                    Either::Left(lhs) => {
                        let lhs: &dyn OutputStream = lhs;
                        lhs
                    }
                    Either::Right(rhs) => {
                        let rhs: &dyn OutputStream = rhs;
                        rhs
                    }
                }
            }
        }
        impl Stream for Either<$lhs, $rhs> {
            fn spawn_layer(&self) -> Layer {
                self.as_stream().spawn_layer()
            }

            fn eval_layer(&self) -> Layer {
                self.as_stream().eval_layer()
            }

            fn name(&self) -> &str {
                self.as_stream().name()
            }

            fn ty(&self) -> &Type {
                self.as_stream().ty()
            }

            fn is_input(&self) -> bool {
                self.as_stream().is_input()
            }

            fn is_parameterized(&self) -> bool {
                self.as_stream().is_parameterized()
            }

            fn is_spawned(&self) -> bool {
                self.as_stream().is_spawned()
            }

            fn is_closed(&self) -> bool {
                self.as_stream().is_closed()
            }

            fn is_eval_filtered(&self) -> bool {
                self.as_stream().is_eval_filtered()
            }

            fn values_to_memorize(&self) -> MemorizationBound {
                self.as_stream().values_to_memorize()
            }

            fn as_stream_ref(&self) -> StreamReference {
                self.as_stream().as_stream_ref()
            }

            fn accessed_by(&self) -> &Accesses {
                self.as_stream().accessed_by()
            }

            fn aggregated_by(&self) -> &[(StreamReference, Origin, WindowReference)] {
                self.as_stream().aggregated_by()
            }

            fn aggregates(&self) -> &[(StreamReference, Origin, WindowReference)] {
                self.as_stream().aggregates()
            }

            fn tags(&self) -> &HashMap<String, Option<String>> {
                self.as_stream().tags()
            }
        }
    };
}

implement_either!(UnparameterizedOutputStream, ParameterizedOutputStream);
implement_either!(ParameterizedOutputStream, UnparameterizedOutputStream);

/// A trigger (represented by the output stream `output_reference`)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Copy)]
pub struct Trigger {
    /// The reference of the output stream representing this trigger
    pub output_reference: OutputReference,
    /// The reference of this trigger
    pub trigger_reference: TriggerReference,
}

impl CommonOutputStream {
    fn is_trigger(&self) -> bool {
        matches!(self.kind, OutputKind::Trigger(_))
    }
}

type Tags = HashMap<String, Option<String>>;

/// A type alias for references to triggers.
pub type TriggerReference = usize;

/// Information on the spawn behavior of a stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spawn {
    /// The timing of when a new instance _could_ be created assuming the spawn condition evaluates to true.
    pub pacing: PacingType,
    /// The spawn condition.  If the condition evaluates to false, the stream will not be spawned.
    pub condition: Option<Expression>,
    #[cfg(feature = "spanned")]
    /// The span of the spawn clause
    pub span: Span,
}

impl Default for Spawn {
    fn default() -> Self {
        Spawn {
            pacing: PacingType::Constant,
            condition: None,
            #[cfg(feature = "spanned")]
            span: Span::Unknown,
        }
    }
}

/// Information on the close behavior of a stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Close {
    /// The `condition` expression needs to be evaluated whenever the stream with this Close template is supposed to be closed.  The result of the evaluation constitutes whether the stream is closed.
    pub condition: Option<Expression>,
    /// The timing of the close condition.
    pub pacing: PacingType,
    /// Indicates whether the close condition contains a reference to the stream it belongs to.
    pub has_self_reference: bool,
    #[cfg(feature = "spanned")]
    /// The span of the close clause
    pub span: Span,
}

impl Default for Close {
    fn default() -> Self {
        Close {
            condition: None,
            pacing: PacingType::Constant,
            has_self_reference: false,
            #[cfg(feature = "spanned")]
            span: Span::Unknown,
        }
    }
}

/// Information on the evaluation behavior of a stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Eval {
    /// The eval clauses of the stream.
    pub clauses: Vec<EvalClause>,
    /// The eval pacing of the stream, combining the condition and expr pacings of all eval clauses
    pub eval_pacing: PacingType,
}

/// Information on an eval clause of a stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalClause {
    /// The expression of this stream needs to be evaluated whenever this condition evaluates to `True`.
    pub condition: Option<Expression>,
    /// The evaluation expression of this stream, defining the returned and accessed value.
    pub expression: Expression,
    /// The eval pacing of the stream, combining the condition and expr pacings of the clause.
    pub pacing: PacingType,
    #[cfg(feature = "spanned")]
    /// The span of the eval clause
    pub span: Span,
}

/// Information of a parameter of a parametrized output stream
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    /// The name of the parameter.
    pub name: String,
    /// The type of the parameter.
    pub ty: Type,
    /// The index of the parameter.
    pub idx: usize,
    #[cfg(feature = "spanned")]
    /// The span of the parameter
    pub span: Span,
}

/// Wrapper for output streams providing additional information specific to time-driven streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct TimeDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
    /// The evaluation frequency of the stream.
    pub frequency: UOM_Frequency,
    /// Whether the given frequency is relative to a dynamic spawn
    pub locality: PacingLocality,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
/// Describes if the pacing is interpreted relatively to a dynamic spawn
pub enum PacingLocality {
    /// The pacing is relative to a global clock
    Global,
    /// The pacing is relative to the spawn
    Local,
}

impl TimeDrivenStream {
    /// Returns the evaluation period, i.e., the multiplicative inverse of [TimeDrivenStream::frequency].
    pub fn period(&self) -> UOM_Time {
        UOM_Time::new::<uom::si::time::second>(
            self.frequency.get::<uom::si::frequency::hertz>().inv(),
        )
    }

    /// Returns the evaluation frequency.
    pub fn frequency(&self) -> UOM_Frequency {
        self.frequency
    }

    /// Returns the evaluation period, i.e., the multiplicative inverse of [TimeDrivenStream::frequency], as [Duration].
    pub fn period_in_duration(&self) -> Duration {
        Duration::from_nanos(
            self.period()
                .get::<nanosecond>()
                .to_integer()
                .try_into()
                .expect("Period [ns] too large for u64!"),
        )
    }
}

/// Wrapper for output streams providing additional information specific to event-based streams.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct EventDrivenStream {
    /// A reference to the stream that is specified
    pub reference: StreamReference,
    /// The activation condition of an event-based stream
    pub ac: ActivationCondition,
}

/// Representation of the activation condition of event-based entities such as streams or spawn conditions
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum ActivationCondition {
    /// Activate when all entries of the [Vec] are true.
    Conjunction(Vec<Self>),
    /// Activate when at least one entry of the [Vec] is true.
    Disjunction(Vec<Self>),
    /// Activate when the referenced stream is evaluated.
    Stream(StreamReference),
    /// Activate
    True,
}

/// Represents an expression
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Expression {
    /// The kind and all kind-specific information of the expression
    pub kind: ExpressionKind,
    /// The type of the expression
    pub ty: Type,
    #[cfg(feature = "spanned")]
    /// The span of the expression
    pub span: Span,
}

/// This enum contains all possible kinds of expressions and their relevant information.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ExpressionKind {
    /// Load a constant value
    LoadConstant(Constant),
    /// Apply an arithmetic or logic operation.  The function is monomorphized.
    ///
    /// *Note:* Arguments never need to be coerced.
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp, Vec<Expression>),
    /// Access another stream
    StreamAccess {
        /// The target stream to be accessed
        target: StreamReference,
        /// The parameters of the specific stream instance that is accessed.  
        ///
        /// If the stream behind `target` is not parametrized, this collection is empty.
        parameters: Vec<Expression>,
        /// The kind of access
        access_kind: StreamAccessKind,
    },
    /// Access to the parameter of a stream represented by a stream reference,
    /// referencing the target stream and the index of the parameter that should be accessed.
    ParameterAccess(StreamReference, usize),
    /// Access to the lambda parameter in the filtered instance aggregation
    LambdaParameterAccess {
        /// Reference to the instance aggregation using the lambda function
        wref: WindowReference,
        /// Reference to the parameter
        pref: usize,
    },
    /// A conditional (if-then-else) expression
    Ite {
        /// The condition under which either `consequence` or `alternative` is selected.
        condition: Box<Expression>,
        /// The consequence should be evaluated and returned if the condition evaluates to true.
        consequence: Box<Expression>,
        /// The alternative should be evaluated and returned if the condition evaluates to false.
        alternative: Box<Expression>,
    },
    /// A tuple expression
    Tuple(Vec<Expression>),
    /// Represents a tuple projections, i.e., it accesses a specific tuple element.  
    // The expression produces a tuple and the `usize` is the index of the accessed element.  This value is constant.
    TupleAccess(Box<Expression>, usize),
    /// Represents a function call.  The function is monomorphized.
    ///
    /// *Note:* Arguments never need to be coerced.
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    Function(String, Vec<Expression>),
    /// Converting a value to a different type
    ///
    /// The result type is indicated in the expression with the `Convert` kind.  
    Convert {
        /// The expression that produces a value.  The type of the expression indicates the source of the conversion.
        expr: Box<Expression>,
    },
    /// Transforms an optional value into a definitive one
    Default {
        /// The expression that results in an optional value.
        expr: Box<Expression>,
        /// An infallible expression providing the default value if `expr` fails to produce a value.
        default: Box<Expression>,
    },
}

/// Represents a constant value of a certain kind.
///
/// *Note* the type of the constant might be more general than the type of the constant.  For example, `Constant::UInt(3u64)` represents an RTLola UInt8 constant.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Constant {
    #[allow(missing_docs)]
    Str(String),
    #[allow(missing_docs)]
    Bool(bool),
    #[allow(missing_docs)]
    UInt(u64),
    #[allow(missing_docs)]
    Int(i64),
    #[allow(missing_docs)]
    Float(f64),
    #[allow(missing_docs)]
    Decimal(Decimal),
}

/// Arithmetical and logical operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithLogOp {
    /// Logic negation (!)
    Not,
    /// Arithmetic negation (-)
    Neg,
    /// Arithmetic addition (+)
    Add,
    /// Arithmetic subtraction (-)
    Sub,
    /// Arithmetic multiplication (*)
    Mul,
    /// Arithmetic division (/)
    Div,
    /// Arithmetic modulation (%)
    Rem,
    /// Arithmetic exponentiation (**)
    Pow,
    /// Logic conjunction/multiplication (&&)
    And,
    /// Logic disjunction/addition (||)
    Or,
    /// Bit-wise xor (^)
    BitXor,
    /// Bit-wise conjunction/multiplication (&)
    BitAnd,
    /// Bit-wise disjunction/addition (|)
    BitOr,
    /// Bit-wise negation / One's complement (~)
    BitNot,
    /// Bit-wise left-shift (<<)
    Shl,
    /// Bit-wise right-shift (>>)
    Shr,
    /// Semantic Equality (==)
    Eq,
    /// Less-than comparison (<)
    Lt,
    /// Less-than-or-equal comparison (<=)
    Le,
    /// Semantic Inequality (!=)
    Ne,
    /// Greater-than-or-equal comparison (>=)
    Ge,
    /// Greater-than comparison (>)
    Gt,
}

/// Represents an instance of a discrete window
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct DiscreteWindow {
    /// The stream whose values will be aggregated
    pub target: StreamReference,
    /// The stream in which expression this window occurs
    pub caller: StreamReference,
    /// The duration over which the window aggregates
    pub duration: usize,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` number of values have been observed.
    pub wait: bool,
    /// The aggregation operation
    pub op: WindowOperation,
    /// A reference to this discrete window
    pub reference: WindowReference,
    /// The type of value the window produces
    pub ty: Type,
    /// The origin of the discrete window expression
    pub origin: Origin,
    /// The pacing of the discrete window expression
    pub pacing: PacingType,
}

/// Represents an instance of a sliding window
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct SlidingWindow {
    /// The stream whose values will be aggregated
    pub target: StreamReference,
    /// The stream in which expression this window occurs
    pub caller: StreamReference,
    /// The duration over which the window aggregates
    pub duration: Duration,
    /// The number of buckets that are needed for the window
    pub num_buckets: MemorizationBound,
    /// The time per bucket of the window
    pub bucket_size: Duration,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once
    pub wait: bool,
    /// The aggregation operation
    pub op: WindowOperation,
    /// A reference to this sliding window
    pub reference: WindowReference,
    /// The type of value the window produces
    pub ty: Type,
    /// The origin of the sliding window expression
    pub origin: Origin,
    /// The pacing of the sliding window expression
    pub pacing: PacingType,
}

/// Represents an instance of an instance aggregation
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct InstanceAggregation {
    /// The stream whose values will be aggregated
    pub target: StreamReference,
    /// The stream calling and evaluating this window
    pub caller: StreamReference,
    /// A filter over the instances
    pub selection: InstanceSelection,
    /// The operation to be performed over the instances
    pub aggr: InstanceOperation,
    /// The reference of this window.
    pub reference: WindowReference,
    /// The type of value the window produces
    pub ty: Type,
    /// The origin of the instance window expression
    pub origin: Origin,
    /// The pacing of the instance aggregation expression
    pub pacing: PacingType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Enum to indicate which instances are part of the aggregation
pub enum InstanceSelection {
    /// Only instances that are updated in this evaluation cycle are part of the aggregation
    Fresh,
    /// All instances are part of the aggregation
    All,
    /// Only instances that are updated in this evaluation cycle and satisfy the condition are part of the aggregation
    FilteredFresh {
        /// The parameters of the lambda expression
        parameters: Vec<Parameter>,
        /// The condition that needs to be satisfied
        cond: Box<Expression>,
    },
    /// All instances that satisfy the condition are part of the aggregation
    FilteredAll {
        /// The parameters of the lambda expression
        parameters: Vec<Parameter>,
        /// The condition that needs to be satisfied
        cond: Box<Expression>,
    },
}

impl InstanceSelection {
    /// Accesses the condition to a filtered instance aggregation. Returns None if the instance aggregation is not filtered
    pub fn condition(&self) -> Option<&Expression> {
        match self {
            InstanceSelection::Fresh | InstanceSelection::All => None,
            InstanceSelection::FilteredFresh {
                parameters: _,
                cond,
            }
            | InstanceSelection::FilteredAll {
                parameters: _,
                cond,
            } => Some(cond),
        }
    }

    /// Accesses the parameters to a filtered instance aggregation. Returns None if the instance aggregation is not filtered
    pub fn parameters(&self) -> Option<&Vec<Parameter>> {
        match self {
            InstanceSelection::Fresh | InstanceSelection::All => None,
            InstanceSelection::FilteredFresh {
                parameters,
                cond: _,
            }
            | InstanceSelection::FilteredAll {
                parameters,
                cond: _,
            } => Some(parameters),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
/// A subset of the window operations that are suitable to be performed over a set of instances.
pub enum InstanceOperation {
    /// Aggregation function to count the number of instances of the accessed stream
    Count,
    /// Aggregation function to return the minimum
    Min,
    /// Aggregation function to return parameter of the minimum
    ArgMin,
    /// Aggregation function to return the minimum
    Max,
    /// Aggregation function to return the parameter of the maximum
    ArgMax,
    /// Aggregation function to return the addition
    Sum,
    /// Aggregation function to return the product
    Product,
    /// Aggregation function to return the average
    Average,
    /// Aggregation function to return the conjunction, i.e., the instances aggregation returns true iff ALL current values of the instances of the accessed stream are assigned to true
    Conjunction,
    /// Aggregation function to return the disjunction, i.e., the instances aggregation returns true iff ANY current values of the instances of the accessed stream are assigned to true
    Disjunction,
    /// Aggregation function to return the variance of all values, assumes equal probability.
    Variance,
    /// Aggregation function to return the covariance of all values in a tuple stream, assumes equal probability.
    Covariance,
    /// Aggregation function to return the standard deviation of all values, assumes equal probability.
    StandardDeviation,
    /// Aggregation function to return the Nth-Percentile
    NthPercentile(u8),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
/// The Ast representation of the different aggregation functions
pub enum WindowOperation {
    /// Aggregation function to count the number of updated values on the accessed stream
    Count,
    /// Aggregation function to return the minimum
    Min,
    /// Aggregation function to return the parameter of the minimum
    ArgMin,
    /// Aggregation function to return the maximum
    Max,
    /// Aggregation function to return the parameter of the maximum
    ArgMax,
    /// Aggregation function to return the addition
    Sum,
    /// Aggregation function to return the product
    Product,
    /// Aggregation function to return the average
    Average,
    /// Aggregation function to return the integral
    Integral,
    /// Aggregation function to return the conjunction, i.e., the sliding window returns true iff ALL values on the accessed stream inside a window are assigned to true
    Conjunction,
    /// Aggregation function to return the disjunction, i.e., the sliding window returns true iff AT LEAst ONE value on the accessed stream inside a window is assigned to true
    Disjunction,
    /// Aggregation function to return the last value, a time bounded hold
    Last,
    /// Aggregation function to return the variance of all values, assumes equal probability.
    Variance,
    /// Aggregation function to return the covariance of all values in a tuple stream, assumes equal probability.
    Covariance,
    /// Aggregation function to return the standard deviation of all values, assumes equal probability.
    StandardDeviation,
    /// Aggregation function to return the Nth-Percentile
    NthPercentile(u8),
}

impl From<InstanceOperation> for WindowOperation {
    fn from(value: InstanceOperation) -> Self {
        match value {
            InstanceOperation::Count => WindowOperation::Count,
            InstanceOperation::Min => WindowOperation::Min,
            InstanceOperation::ArgMin => WindowOperation::ArgMin,
            InstanceOperation::Max => WindowOperation::Max,
            InstanceOperation::ArgMax => WindowOperation::ArgMax,
            InstanceOperation::Sum => WindowOperation::Sum,
            InstanceOperation::Product => WindowOperation::Product,
            InstanceOperation::Average => WindowOperation::Average,
            InstanceOperation::Conjunction => WindowOperation::Conjunction,
            InstanceOperation::Disjunction => WindowOperation::Disjunction,
            InstanceOperation::Variance => WindowOperation::Variance,
            InstanceOperation::Covariance => WindowOperation::Covariance,
            InstanceOperation::StandardDeviation => WindowOperation::StandardDeviation,
            InstanceOperation::NthPercentile(x) => WindowOperation::NthPercentile(x),
        }
    }
}

/// A trait for any kind of window
pub trait Window {
    /// Returns a reference to the stream that will be aggregated by that window.
    fn target(&self) -> StreamReference;

    /// Returns a reference to the stream in which expression this window occurs.
    fn caller(&self) -> StreamReference;

    /// Returns the aggregation operation the window uses.
    fn op(&self) -> WindowOperation;

    /// Returns the type of value the window produces.
    fn ty(&self) -> &Type;

    /// Returns the memorization bound of the window.
    fn memory_bound(&self) -> MemorizationBound;
}

////////// Implementations //////////

impl Stream for UnparameterizedOutputStream {
    fn spawn_layer(&self) -> Layer {
        self.layer.spawn_layer()
    }

    fn eval_layer(&self) -> Layer {
        self.layer.evaluation_layer()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn is_input(&self) -> bool {
        false
    }

    fn is_parameterized(&self) -> bool {
        false
    }

    fn is_spawned(&self) -> bool {
        self.spawn.condition.is_some() || self.spawn.pacing != PacingType::Constant
    }

    fn is_closed(&self) -> bool {
        self.close.condition.is_some()
    }

    fn is_eval_filtered(&self) -> bool {
        self.eval
            .clauses
            .iter()
            .any(|eval| eval.condition.is_some())
    }

    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }

    fn as_stream_ref(&self) -> StreamReference {
        self.reference.sr()
    }

    fn accessed_by(&self) -> &Accesses {
        &self.accessed_by
    }

    fn aggregated_by(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregated_by
    }

    fn aggregates(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregates
    }

    fn tags(&self) -> &HashMap<String, Option<String>> {
        &self.tags
    }

    #[cfg(feature = "spanned")]
    fn tags_span(&self) -> &HashMap<String, Span> {
        &self.tags_span
    }
}

impl Stream for ParameterizedOutputStream {
    fn spawn_layer(&self) -> Layer {
        self.layer.spawn_layer()
    }

    fn eval_layer(&self) -> Layer {
        self.layer.evaluation_layer()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn is_input(&self) -> bool {
        false
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn is_spawned(&self) -> bool {
        true
    }

    fn is_closed(&self) -> bool {
        self.close.condition.is_some()
    }

    fn is_eval_filtered(&self) -> bool {
        self.eval
            .clauses
            .iter()
            .any(|eval| eval.condition.is_some())
    }

    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }

    fn as_stream_ref(&self) -> StreamReference {
        self.reference.sr()
    }

    fn accessed_by(&self) -> &Accesses {
        &self.accessed_by
    }

    fn aggregated_by(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregated_by
    }

    fn aggregates(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregates
    }

    fn tags(&self) -> &HashMap<String, Option<String>> {
        &self.tags
    }

    #[cfg(feature = "spanned")]
    fn tags_span(&self) -> &HashMap<String, Span> {
        &self.tags_span
    }
}

impl Stream for InputStream {
    fn spawn_layer(&self) -> Layer {
        self.layer.spawn_layer()
    }

    fn eval_layer(&self) -> Layer {
        self.layer.evaluation_layer()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn is_input(&self) -> bool {
        true
    }

    fn is_parameterized(&self) -> bool {
        false
    }

    fn is_spawned(&self) -> bool {
        false
    }

    fn is_closed(&self) -> bool {
        false
    }

    fn is_eval_filtered(&self) -> bool {
        false
    }

    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }

    fn as_stream_ref(&self) -> StreamReference {
        StreamReference::In(self.reference)
    }

    fn accessed_by(&self) -> &Accesses {
        &self.accessed_by
    }

    fn aggregated_by(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregated_by
    }

    fn aggregates(&self) -> &[(StreamReference, Origin, WindowReference)] {
        &self.aggregates
    }

    fn tags(&self) -> &HashMap<String, Option<String>> {
        &self.tags
    }

    #[cfg(feature = "spanned")]
    fn tags_span(&self) -> &HashMap<String, Span> {
        &self.tags_span
    }
}

impl Window for SlidingWindow {
    fn target(&self) -> StreamReference {
        self.target
    }

    fn caller(&self) -> StreamReference {
        self.caller
    }

    fn op(&self) -> WindowOperation {
        self.op
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn memory_bound(&self) -> MemorizationBound {
        self.num_buckets
    }
}

impl Window for DiscreteWindow {
    fn target(&self) -> StreamReference {
        self.target
    }

    fn caller(&self) -> StreamReference {
        self.caller
    }

    fn op(&self) -> WindowOperation {
        self.op
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn memory_bound(&self) -> MemorizationBound {
        MemorizationBound::Bounded(self.duration as u32)
    }
}

impl Window for InstanceAggregation {
    fn target(&self) -> StreamReference {
        self.target
    }

    fn caller(&self) -> StreamReference {
        self.caller
    }

    fn op(&self) -> WindowOperation {
        self.aggr.into()
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn memory_bound(&self) -> MemorizationBound {
        MemorizationBound::Bounded(1)
    }
}

impl RtLolaMir {
    /// Returns a collection containing a reference to each input stream in the specification.
    pub fn input_refs(&self) -> impl Iterator<Item = InputReference> {
        0..self.inputs.len()
    }

    /// Returns a collection containing a reference to each output stream in the specification.
    pub fn output_refs(&self) -> impl Iterator<Item = OutputReference> + '_ {
        self.unparameterized_outputs
            .iter()
            .map(|s| s.reference)
            .chain(self.parameterized_outputs.iter().map(|s| s.reference))
    }

    /// Provides mutable access to an input stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::Out].
    pub fn input_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::In(ix) => &mut self.inputs[ix],
            StreamReference::Out(_) => {
                unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`.")
            }
        }
    }

    /// Provides immutable access to an input stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::Out].
    pub fn input(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::In(ix) => &self.inputs[ix],
            StreamReference::Out(_) => {
                unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`.")
            }
        }
    }

    /// Provides mutable access to an output stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::In].
    pub fn output_mut(&mut self, reference: StreamReference) -> &mut dyn OutputStream {
        match reference {
            StreamReference::In(_) => {
                unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`.")
            }
            StreamReference::Out(OutputReference::Unparameterized(idx)) => {
                &mut self.unparameterized_outputs[idx]
            }
            StreamReference::Out(OutputReference::Parameterized(idx)) => {
                &mut self.parameterized_outputs[idx]
            }
        }
    }

    /// Provides immutable access to an output stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::In].
    pub fn output(&self, reference: StreamReference) -> &dyn OutputStream {
        match reference {
            StreamReference::In(_) => {
                unreachable!("Called `LolaIR::output` with a `StreamReference::InRef`.")
            }
            StreamReference::Out(OutputReference::Unparameterized(idx)) => {
                &self.unparameterized_outputs[idx]
            }
            StreamReference::Out(OutputReference::Parameterized(idx)) => {
                &self.parameterized_outputs[idx]
            }
        }
    }

    /// Provides immutable access to a parameterized output stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::In].
    pub fn parameterized_output(&self, reference: StreamReference) -> &ParameterizedOutputStream {
        match reference {
            StreamReference::In(_) => {
                unreachable!(
                    "Called `LolaIR::parameterized_output` with a `StreamReference::InRef`."
                )
            }
            StreamReference::Out(OutputReference::Unparameterized(_)) => {
                unreachable!("Called `LolaIR::parameterized_output` with a unparameterized stream.")
            }
            StreamReference::Out(OutputReference::Parameterized(idx)) => {
                &self.parameterized_outputs[idx]
            }
        }
    }

    /// Returns an iterator over all output streams in the specification (parameterized and unparameterized).
    pub fn outputs(&self) -> impl Iterator<Item = &dyn OutputStream> {
        self.unparameterized_outputs
            .iter()
            .map(|o| {
                let o: &dyn OutputStream = o;
                o
            })
            .chain(self.parameterized_outputs.iter().map(|o| {
                let o: &dyn OutputStream = o;
                o
            }))
    }

    /// Provides immutable access to a stream.
    pub fn stream(&self, reference: StreamReference) -> &dyn Stream {
        match reference {
            StreamReference::In(ix) => &self.inputs[ix],
            StreamReference::Out(_) => self.output(reference),
        }
    }

    /// Produces an iterator over all stream references.
    pub fn all_streams(&self) -> impl Iterator<Item = StreamReference> + '_ {
        self.input_refs()
            .map(StreamReference::In)
            .chain(self.output_refs().map(StreamReference::Out))
    }

    /// Provides a collection of all output streams representing a trigger.
    pub fn all_triggers(&self) -> Vec<&dyn OutputStream> {
        self.triggers
            .iter()
            .map(|t| self.output(t.output_reference.sr()))
            .collect()
    }

    /// Provides a collection of all event-driven output streams.
    pub fn all_event_driven(&self) -> Vec<&dyn OutputStream> {
        self.event_driven
            .iter()
            .map(|t| self.output(t.reference))
            .collect()
    }

    /// Return true if the specification contains any time-driven features.
    /// This includes time-driven streams and time-driven spawn conditions.
    pub fn has_time_driven_features(&self) -> bool {
        !self.time_driven.is_empty()
            || self.outputs().any(|o| {
                matches!(
                    o.spawn.pacing,
                    PacingType::GlobalPeriodic(_) | PacingType::LocalPeriodic(_)
                ) || matches!(
                    o.close.pacing,
                    PacingType::GlobalPeriodic(_) | PacingType::LocalPeriodic(_)
                )
            })
    }

    /// Provides a collection of all time-driven output streams.
    pub fn all_time_driven(&self) -> Vec<&dyn OutputStream> {
        self.time_driven
            .iter()
            .map(|t| self.output(t.reference))
            .collect()
    }

    /// Provides the activation contion of a event-driven stream and none if the stream is time-driven
    pub fn get_ac(&self, sref: StreamReference) -> Option<&ActivationCondition> {
        self.event_driven
            .iter()
            .find(|e| e.reference == sref)
            .map(|e| &e.ac)
    }

    /// Provides immutable access to a discrete window.
    ///
    /// # Panic
    /// Panics if `window` is not a [WindowReference::Discrete].
    pub fn discrete_window(&self, window: WindowReference) -> &DiscreteWindow {
        match window {
            WindowReference::Discrete(x) => &self.discrete_windows[x],
            WindowReference::Sliding(_) | WindowReference::Instance(_) => {
                panic!("wrong type of window reference passed to getter")
            }
        }
    }

    /// Provides immutable access to a instance aggregation.
    ///
    /// # Panic
    /// Panics if `window` is not a [WindowReference::Instance].
    pub fn instance_aggregation(&self, window: WindowReference) -> &InstanceAggregation {
        match window {
            WindowReference::Instance(x) => &self.instance_aggregations[x],
            WindowReference::Sliding(_) | WindowReference::Discrete(_) => {
                panic!("wrong type of window reference passed to getter")
            }
        }
    }

    /// Provides immutable access to a sliding window.
    ///
    /// # Panic
    /// Panics if `window` is not a [WindowReference::Sliding].
    pub fn sliding_window(&self, window: WindowReference) -> &SlidingWindow {
        match window {
            WindowReference::Sliding(x) => &self.sliding_windows[x],
            WindowReference::Discrete(_) | WindowReference::Instance(_) => {
                panic!("wrong type of window reference passed to getter")
            }
        }
    }

    /// Provides immutable access to a window.
    pub fn window(&self, window: WindowReference) -> &dyn Window {
        match window {
            WindowReference::Sliding(x) => &self.sliding_windows[x],
            WindowReference::Discrete(x) => &self.discrete_windows[x],
            WindowReference::Instance(x) => &self.instance_aggregations[x],
        }
    }

    /// Provides a representation for the evaluation layers of all event-driven output streams.  Each element of the outer `Vec` represents a layer, each element of the inner `Vec` an output stream in the layer.
    pub fn get_event_driven_layers(&self) -> Vec<Vec<Task>> {
        let mut event_driven_spawns = self
            .outputs()
            .filter(|o| matches!(o.spawn.pacing, PacingType::Event(_)))
            .peekable();

        // Peekable is fine because the filter above does not have side effects
        if self.event_driven.is_empty() && event_driven_spawns.peek().is_none() {
            return vec![];
        }

        // Zip eval layer with stream reference.
        let streams_with_layers = self.event_driven.iter().map(|s| s.reference).map(|r| {
            (
                self.output(r).eval_layer().into(),
                Task::Evaluate(r.out_ix()),
            )
        });

        let spawns_with_layers =
            event_driven_spawns.map(|o| (o.spawn_layer().inner(), Task::Spawn(o.reference)));

        let tasks_with_layers: Vec<(usize, Task)> =
            streams_with_layers.chain(spawns_with_layers).collect();

        // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
        // layers without entries and more layers than streams.
        // Minimization works as follows:
        // a) Find the greatest layer
        // b) For each potential layer...
        // c) Find streams that would be in it.
        // d) If there is none, skip this layer
        // e) If there are some, add them as layer.

        // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
        let max_layer = tasks_with_layers
            .iter()
            .max_by_key(|(layer, _)| layer)
            .unwrap()
            .0;

        let mut layers = Vec::new();
        // b) For each potential layer
        for i in 0..=max_layer {
            // c) Find streams that would be in it.
            let in_layer_i: Vec<Task> = tasks_with_layers
                .iter()
                .filter_map(|(l, r)| if *l == i { Some(*r) } else { None })
                .collect();
            if in_layer_i.is_empty() {
                // d) If there is none, skip this layer
                continue;
            } else {
                // e) If there are some, add them as layer.
                layers.push(in_layer_i);
            }
        }
        layers
    }

    /// Attempts to compute a schedule for all time-driven streams.
    ///
    /// # Fail
    /// Fails if the resulting schedule would require at least 10^7 deadlines.
    pub fn compute_schedule(&self) -> Result<Schedule, String> {
        Schedule::from(self)
    }

    /// Creates a new [RtLolaMirPrinter] for the Mir type `T`. It implements the [Display](std::fmt::Display) Trait for type `T`.
    pub fn display<'a, T>(&'a self, target: &'a T) -> RtLolaMirPrinter<'a, T> {
        RtLolaMirPrinter::new(self, target)
    }

    /// Represents the specification as a dependency graph
    pub fn dependency_graph(&self) -> DependencyGraph<'_> {
        DependencyGraph::new(self)
    }

    /// Returns the input stream with the given name if it exists.
    pub fn get_input_by_name(&self, name: &str) -> Option<&InputStream> {
        self.inputs.iter().find(|input| input.name == name)
    }

    /// Returns the output stream with the given name if it exists.
    pub fn get_output_by_name(&self, name: &str) -> Option<&dyn OutputStream> {
        self.outputs().find(|output| output.name == name)
    }

    /// Returns the stream with the given name if it exists.
    pub fn get_stream_by_name(&self, name: &str) -> Option<&dyn Stream> {
        self.get_input_by_name(name)
            .map(|input| {
                // clippy likes it this way
                let input: &dyn Stream = input;
                input
            })
            .or_else(|| {
                self.get_output_by_name(name).map(|output| {
                    let output: &dyn Stream = output;
                    output
                })
            })
    }
}

impl Type {
    /// Indicates how many bytes a type requires to be stored in memory.
    ///
    /// Recursive types yield the sum of their sub-type sizes, unsized types panic, and functions do not have a size, so they produce `None`.
    /// # Panics
    /// Panics if the type is an instance of [Type::Option], [Type::String], or [Type::Bytes] because their size is undetermined.
    pub fn size(&self) -> Option<ValSize> {
        match self {
            Type::Bool => Some(ValSize(1)),
            Type::Int(IntTy::Int8) => Some(ValSize(1)),
            Type::Int(IntTy::Int16) => Some(ValSize(2)),
            Type::Int(IntTy::Int32) => Some(ValSize(4)),
            Type::Int(IntTy::Int64) => Some(ValSize(8)),
            Type::Int(IntTy::Int128) => Some(ValSize(16)),
            Type::Int(IntTy::Int256) => Some(ValSize(32)),
            Type::UInt(UIntTy::UInt8) => Some(ValSize(1)),
            Type::UInt(UIntTy::UInt16) => Some(ValSize(2)),
            Type::UInt(UIntTy::UInt32) => Some(ValSize(4)),
            Type::UInt(UIntTy::UInt64) => Some(ValSize(8)),
            Type::UInt(UIntTy::UInt128) => Some(ValSize(16)),
            Type::UInt(UIntTy::UInt256) => Some(ValSize(32)),
            Type::Float(FloatTy::Float32) => Some(ValSize(4)),
            Type::Float(FloatTy::Float64) => Some(ValSize(8)),
            Type::Fixed(FixedTy::Fixed64_32) | Type::UFixed(FixedTy::Fixed64_32) => {
                Some(ValSize(64))
            }
            Type::Fixed(FixedTy::Fixed32_16) | Type::UFixed(FixedTy::Fixed32_16) => {
                Some(ValSize(32))
            }
            Type::Fixed(FixedTy::Fixed16_8) | Type::UFixed(FixedTy::Fixed16_8) => Some(ValSize(16)),
            Type::Option(_) => unimplemented!("Size of option not determined, yet."),
            Type::Tuple(t) => {
                let size = t.iter().map(|t| Type::size(t).unwrap().0).sum();
                Some(ValSize(size))
            }
            Type::String | Type::Bytes => unimplemented!("Size of Strings not determined, yet."),
            Type::Function { .. } => None,
            Type::Vec2 => Some(ValSize(Type::Float(FloatTy::Float64).size().unwrap().0 * 2)),
            Type::Vec3 => Some(ValSize(Type::Float(FloatTy::Float64).size().unwrap().0 * 3)),
        }
    }
}

/// The size of a specific value in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ValSize(pub u32); // Needs to be reasonably large for compound types.

impl From<u8> for ValSize {
    fn from(val: u8) -> ValSize {
        ValSize(u32::from(val))
    }
}

impl std::ops::Add for ValSize {
    type Output = ValSize;

    fn add(self, rhs: ValSize) -> ValSize {
        ValSize(self.0 + rhs.0)
    }
}

/// Representation of the different stream accesses
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub enum StreamAccessKind {
    /// Represents the synchronous access
    Sync,
    /// Represents the access to a (discrete window)[DiscreteWindow]
    ///
    /// The argument contains the reference to the (discrete window)[DiscreteWindow] whose value is used in the [Expression].
    DiscreteWindow(WindowReference),
    /// Represents the access to a (sliding window)[SlidingWindow]
    ///
    /// The argument contains the reference to the (sliding window)[SlidingWindow] whose value is used in the [Expression].
    SlidingWindow(WindowReference),
    /// Represents the access to a (instance aggregation)[InstanceAggregation]
    ///
    /// The argument contains the reference to the (instance aggregation)[InstanceAggregation] whose value is used in the [Expression].
    InstanceAggregation(WindowReference),
    /// Representation of sample and hold accesses
    Hold,
    /// Representation of offset accesses
    ///
    /// The argument contains the [Offset] of the stream access.
    Offset(Offset),
    /// Represents the optional `get` access.
    Get,
    /// Represents the update check of a stream, if the target received a new value at this timestamp.
    Fresh,
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub enum Offset {
    /// A strictly positive discrete offset, e.g., `4`, or `42`
    Future(u32),
    /// A non-negative discrete offset, e.g., `0`, `-4`, or `-42`
    Past(u32),
}

impl PartialOrd for Offset {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        use Offset::*;
        match (self, other) {
            (Past(_), Future(_)) => Ordering::Less,
            (Future(_), Past(_)) => Ordering::Greater,
            (Future(a), Future(b)) => a.cmp(b),
            (Past(a), Past(b)) => b.cmp(a),
        }
    }
}

impl Type {
    /// Returns the inner type of an option or itself
    pub fn inner_ty(&self) -> &Type {
        match self {
            Type::Option(inner) => inner.inner_ty(),
            other => other,
        }
    }
}
