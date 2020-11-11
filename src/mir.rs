/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.
*/

use crate::common_ir::*;

pub(crate) mod lowering;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
pub use crate::common_ir::{EventDrivenStream, Layer, StreamReference, TimeDrivenStream, Trigger, WindowReference};
pub use crate::mir::schedule::{Deadline, Schedule};
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for IR

use std::time::Duration;

pub(crate) type Mir = RTLolaMIR;

#[derive(Debug, Clone, PartialEq)]
pub struct RTLolaMIR {
    /// All input streams.
    pub inputs: Vec<InputStream>,
    /// All output streams with the bare minimum of information.
    pub outputs: Vec<OutputStream>,
    /// References to all time-driven streams.
    pub time_driven: Vec<TimeDrivenStream>,
    /// References to all event-driven streams.
    pub event_driven: Vec<EventDrivenStream>,
    /// A collection of all discrete windows.
    pub discrete_windows: Vec<DiscreteWindow>,
    /// A collection of all sliding windows.
    pub sliding_windows: Vec<SlidingWindow>,
    /// A collection of triggers
    pub triggers: Vec<Trigger>,
}

/// Represents a value type. Stream types are no longer relevant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// A binary type
    Bool,
    /// An integer type containing an enum stating its bit-width.
    Int(IntTy),
    /// An unsigned integer type containing an enum stating its bit-width.
    UInt(UIntTy),
    /// An floating point number type containing an enum stating its bit-width.
    Float(FloatTy),
    /// A unicode string
    String,
    /// A sequence of 8bit bytes
    Bytes,
    /// A n-ary tuples where n is the length of the contained vector.
    Tuple(Vec<Type>),
    /// An optional value type, e.g., resulting from accessing a stream with offset -1
    Option(Box<Type>),
    /// A type describing a function containing its argument types and return type. Resolve ambiguities in polymorphic functions and operations.
    Function(Vec<Type>, Box<Type>),
}

impl From<&ValueTy> for Type {
    fn from(ty: &ValueTy) -> Type {
        match ty {
            ValueTy::Bool => Type::Bool,
            ValueTy::Int(i) => Type::Int(*i),
            ValueTy::UInt(u) => Type::UInt(*u),
            ValueTy::Float(f) => Type::Float(*f),
            ValueTy::String => Type::String,
            ValueTy::Bytes => Type::Bytes,
            ValueTy::Tuple(t) => Type::Tuple(t.iter().map(|e| e.into()).collect()),
            ValueTy::Option(o) => Type::Option(Box::new(o.as_ref().into())),
            _ => unreachable!("cannot lower `ValueTy` {}", ty),
        }
    }
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InputStream {
    /// The name of the stream.
    pub name: String,
    /// The type of the stream.
    pub ty: Type,
    /// What streams depend, i.e., access values of this stream.
    pub dependent_streams: Vec<StreamReference>,
    // *was:* pub dependent_streams: Vec<Tracking>,
    /// Which sliding windows aggregate values of this stream.
    pub dependent_windows: Vec<(StreamReference, WindowReference)>,
    // *was:* pub dependent_windows: Vec<WindowReference>,
    /// Indicates in which evaluation layer the stream is.  
    pub layer: StreamLayers,
    /// The amount of memory required for this stream.
    pub memory_bound: MemorizationBound,
    /// The reference pointing to this stream.
    pub reference: StreamReference,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, PartialEq, Clone)]
pub struct OutputStream {
    /// The name of the stream.
    pub name: String,
    /// The type of the stream.
    pub ty: Type,
    /// The stream expression
    pub expr: Expression,
    /// The input streams on which this stream depends.
    pub input_dependencies: Vec<StreamReference>,
    /// The output streams on which this stream depends.
    pub outgoing_dependencies: Vec<StreamReference>,
    /// The Tracking of all streams that depend on this stream.
    pub dependent_streams: Vec<Tracking>,
    /// The sliding windows depending on this stream.
    pub dependent_windows: Vec<WindowReference>,
    /// The amount of memory required for this stream.
    pub memory_bound: MemorizationBound,
    /// Indicates in which evaluation layer the stream is.  
    pub layer: StreamLayers,
    /// The reference pointing to this stream.
    pub reference: StreamReference,
}

/// Represents an expression.
#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    /// The kind of expression.
    pub kind: ExpressionKind,
    /// The type of the expression.
    pub ty: Type,
}

/// The expressions of the IR.
#[derive(Debug, PartialEq, Clone)]
pub enum ExpressionKind {
    /// Loading a constant
    LoadConstant(Constant),
    /// Applying arithmetic or logic operation and its monomorphic type
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp, Vec<Expression>, Type),
    /// Accessing another stream with a potentially 0 offset
    /// 1st argument -> default
    OffsetLookup {
        /// The target of the lookup.
        target: StreamReference,
        /// The offset of the lookup.
        offset: Offset,
    },
    /// Accessing another stream
    StreamAccess(StreamReference, StreamAccessKind),
    /// A discrete window expression over a duration
    DiscreteWindowLookup(WindowReference),
    /// A sliding window expression over a duration
    WindowLookup(WindowReference),
    /// An if-then-else expression
    Ite {
        #[allow(missing_docs)]
        condition: Box<Expression>,
        #[allow(missing_docs)]
        consequence: Box<Expression>,
        #[allow(missing_docs)]
        alternative: Box<Expression>,
    },
    /// A tuple expression
    Tuple(Vec<Expression>),
    /// Represents an access to a specific tuple element.  The second argument indicates the index of the accessed element while the first produces the accessed tuple.
    TupleAccess(Box<Expression>, usize),
    /// A function call with its monomorphic type
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    Function(String, Vec<Expression>, Type),
    /// Converting a value to a different type
    Convert {
        /// The original type
        from: Type,
        /// The target type
        to: Type,
        /// The expression that produces a value of type `from` which should be converted to `to`.
        expr: Box<Expression>,
    },
    /// Transforms an optional value into a "normal" one
    Default {
        /// The expression that results in an optional value.
        expr: Box<Expression>,
        /// An infallible expression providing a default value of `expr` evaluates to `None`.
        default: Box<Expression>,
    },
}

/// Represents a constant value of a certain kind.
#[derive(Debug, PartialEq, Clone)]
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

/// Contains all arithmetical and logical operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithLogOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `**` operator (power)
    Pow,
    /// The `&&` operator (logical and)
    And,
    /// The `||` operator (logical or)
    Or,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `~` operator for one's complement
    BitNot,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

/// Represents an instance of a discrete window.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DiscreteWindow {
    /// The stream whose values will be aggregated.
    pub target: StreamReference,
    /// The duration (number of events) over which the window aggregates.
    pub duration: u64,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
    /// A reference to this discrete window.
    pub reference: WindowReference,
    /// The type of value the window produces.
    pub ty: Type,
}

/// Represents an instance of a sliding window.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SlidingWindow {
    /// The stream whose values will be aggregated.
    pub target: StreamReference,
    /// The duration over which the window aggregates.
    pub duration: Duration,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
    /// A reference to this sliding window.
    pub reference: WindowReference,
    /// The type of value the window produces.
    pub ty: Type,
}

impl Stream for OutputStream {
    fn spawn_layer(&self) -> Layer {
        self.layer.spawn_layer()
    }
    fn eval_layer(&self) -> Layer {
        self.layer.evaluation_layer()
    }
    fn is_input(&self) -> bool {
        false
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
    fn is_input(&self) -> bool {
        true
    }
    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }
    fn as_stream_ref(&self) -> StreamReference {
        self.reference
    }
}

impl RTLolaMIR {
    /// Returns a `Vec` containing a reference for each input stream in the specification.
    pub fn input_refs(&self) -> Vec<InputReference> {
        (0..self.inputs.len()).collect()
    }

    /// Returns a `Vec` containing a reference for each output stream in the specification.
    pub fn output_refs(&self) -> Vec<OutputReference> {
        (0..self.outputs.len()).collect()
    }

    /// Provides mutable access to an input stream.
    pub fn get_in_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::InRef(ix) => &mut self.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides immutable access to an input stream.
    pub fn get_in(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::InRef(ix) => &self.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides mutable access to an output stream.
    pub fn get_out_mut(&mut self, reference: StreamReference) -> &mut OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &mut self.outputs[ix],
        }
    }

    /// Provides immutable access to an output stream.
    pub fn get_out(&self, reference: StreamReference) -> &OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &self.outputs[ix],
        }
    }

    /// Returns a `Vec` containing a reference for each stream in the specification.
    pub fn all_streams(&self) -> Vec<StreamReference> {
        self.input_refs()
            .iter()
            .map(|ix| StreamReference::InRef(*ix))
            .chain(self.output_refs().iter().map(|ix| StreamReference::OutRef(*ix)))
            .collect()
    }

    /// Returns a `Vec` containing a reference to an output stream representing a trigger in the specification.
    pub fn get_triggers(&self) -> Vec<&OutputStream> {
        // self.triggers.iter().map(|t| self.get_out(t.reference)).collect()
        todo!()
    }

    /// Returns a `Vec` containing a reference for each event-driven output stream in the specification.
    pub fn get_event_driven(&self) -> Vec<&OutputStream> {
        // self.event_driven.iter().map(|t| self.get_out(t.reference)).collect()
        todo!()
    }

    /// Returns a `Vec` containing a reference for each time-driven output stream in the specification.
    pub fn get_time_driven(&self) -> Vec<&OutputStream> {
        // self.time_driven.iter().map(|t| self.get_out(t.reference)).collect()
        todo!()
    }

    /// Returns a discrete Window instance for a given WindowReference in the specification
    pub fn get_discrete_window(&self, _window: WindowReference) -> &DiscreteWindow {
        // match window {
        //     WindowReference::DiscreteWindow(x) => &self.discrete_windows[x],
        //     WindowReference::SlidingWindow(_) => panic!("wrong type of window reference passed to getter"),
        // }
        todo!()
    }

    /// Returns a sliding window instance for a given WindowReference in the specification
    pub fn get_window(&self, window: WindowReference) -> &SlidingWindow {
        &self.sliding_windows[window.idx()]
    }

    /// Provides a representation for the evaluation layers of all event-driven output streams.  Each element of the outer `Vec` represents a layer, each element of the inner `Vec` a stream in the layer.
    pub fn get_event_driven_layers(&self) -> Vec<Vec<OutputReference>> {
        if self.event_driven.is_empty() {
            return vec![];
        }

        // Zip eval layer with stream reference.
        let streams_with_layers: Vec<(usize, OutputReference)> = self
            .event_driven
            .iter()
            .map(|s| s.reference)
            .map(|r| (self.get_out(r).eval_layer().into(), r.out_ix()))
            .collect();

        // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
        // layers without entries and more layers than streams.
        // Minimization works as follows:
        // a) Find the greatest layer
        // b) For each potential layer...
        // c) Find streams that would be in it.
        // d) If there is none, skip this layer
        // e) If there are some, add them as layer.

        // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
        let max_layer = streams_with_layers.iter().max_by_key(|(layer, _)| layer).unwrap().0;

        let mut layers = Vec::new();
        // b) For each potential layer
        for i in 0..=max_layer {
            // c) Find streams that would be in it.
            let in_layer_i: Vec<OutputReference> =
                streams_with_layers.iter().filter_map(|(l, r)| if *l == i { Some(*r) } else { None }).collect();
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

    /// Computes a schedule for all time-driven streams.
    pub fn compute_schedule(&self) -> Result<Schedule, String> {
        // TODO adapt to mir
        unimplemented!();
        // Schedule::from(self)
    }
}

impl Type {
    /// Indicates how many bytes a type requires to be stored in memory.
    pub fn size(&self) -> Option<ValSize> {
        match self {
            Type::Bool => Some(ValSize(1)),
            Type::Int(IntTy::I8) => Some(ValSize(1)),
            Type::Int(IntTy::I16) => Some(ValSize(2)),
            Type::Int(IntTy::I32) => Some(ValSize(4)),
            Type::Int(IntTy::I64) => Some(ValSize(8)),
            Type::UInt(UIntTy::U8) => Some(ValSize(1)),
            Type::UInt(UIntTy::U16) => Some(ValSize(2)),
            Type::UInt(UIntTy::U32) => Some(ValSize(4)),
            Type::UInt(UIntTy::U64) => Some(ValSize(8)),
            Type::Float(FloatTy::F16) => Some(ValSize(2)),
            Type::Float(FloatTy::F32) => Some(ValSize(4)),
            Type::Float(FloatTy::F64) => Some(ValSize(8)),
            Type::Option(_) => unimplemented!("Size of option not determined, yet."),
            Type::Tuple(t) => {
                let size = t.iter().map(|t| Type::size(t).unwrap().0).sum();
                Some(ValSize(size))
            }
            Type::String | Type::Bytes => unimplemented!("Size of Strings not determined, yet."),
            Type::Function(_, _) => None,
        }
    }
}
