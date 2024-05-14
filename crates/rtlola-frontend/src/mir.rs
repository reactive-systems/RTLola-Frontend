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

use std::convert::TryInto;
use std::time::Duration;

use num::traits::Inv;
pub use print::RtLolaMirPrinter;
use rtlola_hir::hir::ConcreteValueType;
pub use rtlola_hir::hir::{
    InputReference, Layer, MemorizationBound, Origin, OutputKind, OutputReference, StreamLayers, StreamReference,
    WindowReference,
};
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RtLolaMir {
    /// Contains all input streams.
    pub inputs: Vec<InputStream>,
    /// Contains all output streams including all triggers.  They only contain the information relevant for every single kind of output stream.  Refer to [RtLolaMir::time_driven], [RtLolaMir::event_driven],
    /// and [RtLolaMir::triggers] for more information.
    pub outputs: Vec<OutputStream>,
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
}

/// Represents an RTLola value type.  This does not including pacing information, for this refer to [TimeDrivenStream] and [EventDrivenStream].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    /// A boolean type
    Bool,
    /// An integer type of fixed bit-width
    Int(IntTy),
    /// An unsigned integer type of fixed bit-width
    UInt(UIntTy),
    /// A floating point type of fixed bit-width
    Float(FloatTy),
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
}

/// Represents an RTLola pacing type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacingType {
    /// Represents a periodic pacing with a fixed frequency
    Periodic(UOM_Frequency),
    /// Represents an event based pacing defined by an [ActivationCondition]
    Event(ActivationCondition),
    /// The pacing is constant, meaning that the value is always present.
    Constant,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntTy {
    /// Represents an 8-bit integer.
    Int8,
    /// Represents a 16-bit integer.
    Int16,
    /// Represents a 32-bit integer.
    Int32,
    /// Represents a 64-bit integer.
    Int64,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UIntTy {
    /// Represents an 8-bit unsigned integer.
    UInt8,
    /// Represents a 16-bit unsigned integer.
    UInt16,
    /// Represents a 32-bit unsigned integer.
    UInt32,
    /// Represents a 64-bit unsigned integer.
    UInt64,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatTy {
    /// Represents a 32-bit floating point number.
    Float32,
    /// Represents a 64-bit floating point number.
    Float64,
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

/// Contains all information inherent to an input stream.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct InputStream {
    /// The name of the stream
    pub name: String,
    /// The value type of the stream.  Note that its pacing is always pre-determined.
    pub ty: Type,
    /// The collection of streams that access the current stream non-transitively
    pub accessed_by: Vec<(StreamReference, Vec<(Origin, StreamAccessKind)>)>,
    /// The collection of sliding windows that access this stream non-transitively.  This includes both sliding and discrete windows.
    pub aggregated_by: Vec<(StreamReference, WindowReference)>,
    /// Provides the evaluation of layer of this stream.
    pub layer: StreamLayers,
    /// Provides the number of values of this stream's type that need to be memorized.  Refer to [Type::size] to get a type's byte-size.
    pub memory_bound: MemorizationBound,
    /// The reference referring to this stream
    pub reference: StreamReference,
}

/// Contains all information relevant to every kind of output stream.
///
/// Refer to [TimeDrivenStream], [EventDrivenStream], and [Trigger], as well as their respective fields in the Mir for additional information.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct OutputStream {
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
    pub accesses: Vec<(StreamReference, Vec<(Origin, StreamAccessKind)>)>,
    /// The collection of streams that access the current stream non-transitively
    pub accessed_by: Vec<(StreamReference, Vec<(Origin, StreamAccessKind)>)>,
    /// The collection of sliding windows that access this stream non-transitively.  This includes both sliding and discrete windows.
    pub aggregated_by: Vec<(StreamReference, WindowReference)>,
    /// Provides the number of values of this stream's type that need to be memorized.  Refer to [Type::size] to get a type's byte-size.
    pub memory_bound: MemorizationBound,
    /// Provides the evaluation of layer of this stream.
    pub layer: StreamLayers,
    /// The reference referring to this stream
    pub reference: StreamReference,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Parameter>,
}

/// A trigger (represented by the output stream `output_reference`)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Copy)]
pub struct Trigger {
    /// The reference of the output stream representing this trigger
    pub output_reference: StreamReference,
    /// The reference of this trigger
    pub trigger_reference: TriggerReference,
}

impl OutputStream {
    fn is_trigger(&self) -> bool {
        matches!(self.kind, OutputKind::Trigger(_))
    }
}

/// A type alias for references to triggers.
pub type TriggerReference = usize;

/// Information on the spawn behavior of a stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spawn {
    /// The expression needs to be evaluated whenever the stream with this Spawn template is supposed to be spawned.  The result of the evaluation constitutes the respective parameters.
    pub expression: Option<Expression>,
    /// The timing of when a new instance _could_ be created assuming the spawn condition evaluates to true.
    pub pacing: PacingType,
    /// The spawn condition.  If the condition evaluates to false, the stream will not be spawned.
    pub condition: Option<Expression>,
}

impl Default for Spawn {
    fn default() -> Self {
        Spawn {
            expression: None,
            pacing: PacingType::Constant,
            condition: None,
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
}

impl Default for Close {
    fn default() -> Self {
        Close {
            condition: None,
            pacing: PacingType::Constant,
            has_self_reference: false,
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
}

/// Wrapper for output streams providing additional information specific to time-driven streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct TimeDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
    /// The evaluation frequency of the stream.
    pub frequency: UOM_Frequency,
}

impl TimeDrivenStream {
    /// Returns the evaluation period, i.e., the multiplicative inverse of [TimeDrivenStream::frequency].
    pub fn period(&self) -> UOM_Time {
        UOM_Time::new::<uom::si::time::second>(self.frequency.get::<uom::si::frequency::hertz>().inv())
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
}

/// Represents an instance of an instance aggregation
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
/// Enum to indicate which instances are part of the aggregation
pub enum InstanceSelection {
    /// Only instances that are updated in this evaluation cycle are part of the aggregation
    Fresh,
    /// All instances are part of the aggregation
    All,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
/// A subset of the window operations that are suitable to be performed over a set of instances.
pub enum InstanceOperation {
    /// Aggregation function to count the number of instances of the accessed stream
    Count,
    /// Aggregation function to return the minimum
    Min,
    /// Aggregation function to return the minimum
    Max,
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
    /// Aggregation function to return the minimum
    Max,
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
            InstanceOperation::Max => WindowOperation::Max,
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
impl Stream for OutputStream {
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
        self.spawn.expression.is_some()
    }

    fn is_spawned(&self) -> bool {
        self.spawn.expression.is_some() || self.spawn.condition.is_some()
    }

    fn is_closed(&self) -> bool {
        self.close.condition.is_some()
    }

    fn is_eval_filtered(&self) -> bool {
        self.eval.clauses.iter().any(|eval| eval.condition.is_some())
    }

    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }

    fn as_stream_ref(&self) -> StreamReference {
        self.reference
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
        self.reference
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
    pub fn output_refs(&self) -> impl Iterator<Item = OutputReference> {
        0..self.outputs.len()
    }

    /// Provides mutable access to an input stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::Out].
    pub fn input_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::In(ix) => &mut self.inputs[ix],
            StreamReference::Out(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides immutable access to an input stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::Out].
    pub fn input(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::In(ix) => &self.inputs[ix],
            StreamReference::Out(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides mutable access to an output stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::In].
    pub fn output_mut(&mut self, reference: StreamReference) -> &mut OutputStream {
        match reference {
            StreamReference::In(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::Out(ix) => &mut self.outputs[ix],
        }
    }

    /// Provides immutable access to an output stream.
    ///
    /// # Panic
    /// Panics if `reference` is a [StreamReference::In].
    pub fn output(&self, reference: StreamReference) -> &OutputStream {
        match reference {
            StreamReference::In(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::Out(ix) => &self.outputs[ix],
        }
    }

    /// Provides immutable access to a stream.
    pub fn stream(&self, reference: StreamReference) -> &dyn Stream {
        match reference {
            StreamReference::In(ix) => &self.inputs[ix],
            StreamReference::Out(ix) => &self.outputs[ix],
        }
    }

    /// Produces an iterator over all stream references.
    pub fn all_streams(&self) -> impl Iterator<Item = StreamReference> {
        self.input_refs()
            .map(StreamReference::In)
            .chain(self.output_refs().map(StreamReference::Out))
    }

    /// Provides a collection of all output streams representing a trigger.
    pub fn all_triggers(&self) -> Vec<&OutputStream> {
        self.triggers.iter().map(|t| self.output(t.output_reference)).collect()
    }

    /// Provides a collection of all event-driven output streams.
    pub fn all_event_driven(&self) -> Vec<&OutputStream> {
        self.event_driven.iter().map(|t| self.output(t.reference)).collect()
    }

    /// Return true if the specification contains any time-driven features.
    /// This includes time-driven streams and time-driven spawn conditions.
    pub fn has_time_driven_features(&self) -> bool {
        !self.time_driven.is_empty()
            || self
                .outputs
                .iter()
                .any(|o| matches!(o.spawn.pacing, PacingType::Periodic(_)))
    }

    /// Provides a collection of all time-driven output streams.
    pub fn all_time_driven(&self) -> Vec<&OutputStream> {
        self.time_driven.iter().map(|t| self.output(t.reference)).collect()
    }

    /// Provides the activation contion of a event-driven stream and none if the stream is time-driven
    pub fn get_ac(&self, sref: StreamReference) -> Option<&ActivationCondition> {
        self.event_driven.iter().find(|e| e.reference == sref).map(|e| &e.ac)
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
            },
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
            },
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
            },
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
            .outputs
            .iter()
            .filter(|o| matches!(o.spawn.pacing, PacingType::Event(_)))
            .peekable();

        // Peekable is fine because the filter above does not have side effects
        if self.event_driven.is_empty() && event_driven_spawns.peek().is_none() {
            return vec![];
        }

        // Zip eval layer with stream reference.
        let streams_with_layers = self
            .event_driven
            .iter()
            .map(|s| s.reference)
            .map(|r| (self.output(r).eval_layer().into(), Task::Evaluate(r.out_ix())));

        let spawns_with_layers =
            event_driven_spawns.map(|o| (o.spawn_layer().inner(), Task::Spawn(o.reference.out_ix())));

        let tasks_with_layers: Vec<(usize, Task)> = streams_with_layers.chain(spawns_with_layers).collect();

        // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
        // layers without entries and more layers than streams.
        // Minimization works as follows:
        // a) Find the greatest layer
        // b) For each potential layer...
        // c) Find streams that would be in it.
        // d) If there is none, skip this layer
        // e) If there are some, add them as layer.

        // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
        let max_layer = tasks_with_layers.iter().max_by_key(|(layer, _)| layer).unwrap().0;

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
            Type::UInt(UIntTy::UInt8) => Some(ValSize(1)),
            Type::UInt(UIntTy::UInt16) => Some(ValSize(2)),
            Type::UInt(UIntTy::UInt32) => Some(ValSize(4)),
            Type::UInt(UIntTy::UInt64) => Some(ValSize(8)),
            Type::Float(FloatTy::Float32) => Some(ValSize(4)),
            Type::Float(FloatTy::Float64) => Some(ValSize(8)),
            Type::Option(_) => unimplemented!("Size of option not determined, yet."),
            Type::Tuple(t) => {
                let size = t.iter().map(|t| Type::size(t).unwrap().0).sum();
                Some(ValSize(size))
            },
            Type::String | Type::Bytes => unimplemented!("Size of Strings not determined, yet."),
            Type::Function { .. } => None,
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
        use std::cmp::Ordering;

        use Offset::*;
        match (self, other) {
            (Past(_), Future(_)) => Some(Ordering::Less),
            (Future(_), Past(_)) => Some(Ordering::Greater),
            (Future(a), Future(b)) => Some(a.cmp(b)),
            (Past(a), Past(b)) => Some(b.cmp(a)),
        }
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
