/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occures in different modes, adding different information to the intermediate representation.
*/

use crate::analysis::Report;
use crate::ast::RTLolaAst;
use crate::common_ir::*;
// use crate::hir::lowering::Lowering;

pub(crate) mod lowering;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
// pub use crate::hir::schedule::{Deadline, Schedule};
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for IR

use std::time::Duration;

/// Intermediate representation of an RTLola specification.
/// Contains all relevant information found in the underlying specification and is enriched with information collected in semantic analyses.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RTLolaHIR<MODE: HirModes> {
    mode: MODE,
}

impl<MODE: HirModes> RTLolaHIR<MODE> {
    pub(crate) fn new(ast: &RTLolaAst, analysis_result: &Report) -> RTLolaHIR<FullInformationHirMode> {
        // Lowering::new(ast, analysis_result).lower()
        todo!()
    }
}

// TODO currently only the FullInformationHirMode Mode is supported
/// Modes of the intermediate representation
pub(crate) trait HirModes {}
struct RawHirMode;
impl HirModes for RawHirMode {}
struct TypeCheckedHirMode;
impl HirModes for TypeCheckedHirMode {}

/// Intermediate representation with full information of an RTLola specification.
/// Contains all relevant information found in the underlying specification and is enriched with information collected in semantic analyses.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FullInformationHirMode {
    /// All input streams.
    inputs: Vec<InputStream>,
    /// All output streams with the bare minimum of information.
    outputs: Vec<OutputStream>,
    /// References to all time-driven streams.
    time_driven: Vec<TimeDrivenStream>,
    /// References to all event-driven streams.
    event_driven: Vec<EventDrivenStream>,
    /// A collection of all sliding windows.
    sliding_windows: Vec<SlidingWindow>,
    /// A collection of triggers
    triggers: Vec<Trigger>,
}
impl HirModes for FullInformationHirMode {}

/// Represents a value type. Stream types are no longer relevant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Type {
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
pub(crate) struct InputStream {
    /// The name of the stream.
    name: String,
    /// The type of the stream.
    ty: Type,
    /// What streams depend, i.e., access values of this stream.
    dependent_streams: Vec<Tracking>,
    /// Which sliding windows aggregate values of this stream.
    dependent_windows: Vec<WindowReference>,
    /// Indicates in which evaluation layer the stream is.  
    layer: u32,
    /// The amount of memory required for this stream.
    memory_bound: MemorizationBound,
    /// The reference pointing to this stream.
    reference: StreamReference,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct OutputStream {
    /// The name of the stream.
    name: String,
    /// The type of the stream.
    ty: Type,
    /// The stream expression
    expr: Expression,
    /// The input streams on which this stream depends.
    input_dependencies: Vec<StreamReference>,
    /// The output streams on which this stream depends.
    outgoing_dependencies: Vec<Dependency>,
    /// The Tracking of all streams that depend on this stream.
    dependent_streams: Vec<Tracking>,
    /// The sliding windows depending on this stream.
    dependent_windows: Vec<WindowReference>,
    /// The amount of memory required for this stream.
    memory_bound: MemorizationBound,
    /// Indicates in which evaluation layer the stream is.  
    layer: u32,
    /// The reference pointing to this stream.
    reference: StreamReference,
    /// The activation condition, which indicates when this stream needs to be evaluated.  Will be empty if the stream has a fixed frequency.
    ac: Option<Activation<StreamReference>>,
}

/// Represents an expression.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Expression {
    /// The kind of expression.
    pub(crate) kind: ExpressionKind,
    /// The type of the expression.
    pub(crate) ty: Type,
}

/// The expressions of the IR.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ExpressionKind {
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
    /// A window expression over a duration
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
pub(crate) enum Constant {
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
pub(crate) enum ArithLogOp {
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

/// Represents an instance of a sliding window.
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct SlidingWindow {
    /// The stream whose values will be aggregated.
    target: StreamReference,
    /// The duration over which the window aggregates.
    duration: Duration,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    wait: bool,
    /// The aggregation operation.
    op: WindowOperation,
    /// A reference to this sliding window.
    reference: WindowReference,
    /// The type of value the window produces.
    ty: Type,
}

////////// Implementations //////////

impl Stream for OutputStream {
    fn eval_layer(&self) -> u32 {
        self.layer
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
    fn eval_layer(&self) -> u32 {
        self.layer
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

impl Expression {
    fn new(kind: ExpressionKind, ty: Type) -> Self {
        Self { kind, ty }
    }
}

impl RTLolaHIR<FullInformationHirMode> {
    /// Returns a `Vec` containing a reference for each input stream in the specification.
    pub(crate) fn input_refs(&self) -> Vec<InputReference> {
        (0..self.mode.inputs.len()).collect()
    }

    /// Provides mutable access to an input stream.
    pub(crate) fn input_as_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::InRef(ix) => &mut self.mode.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides immutable access to an input stream.
    pub(crate) fn input(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::InRef(ix) => &self.mode.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Returns a mutable `Vec` with the input streams in the specification.
    pub(crate) fn inputs_as_mut<'a>(&'a mut self) -> &'a mut Vec<InputStream> {
        &mut self.mode.inputs
    }

    /// Returns a `Vec` containing a reference for each output stream in the specification.
    pub(crate) fn output_refs(&self) -> Vec<OutputReference> {
        (0..self.mode.outputs.len()).collect()
    }

    /// Provides mutable access to an output stream.
    pub(crate) fn output_as_mut(&mut self, reference: StreamReference) -> &mut OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &mut self.mode.outputs[ix],
        }
    }

    /// Provides immutable access to an output stream.
    pub(crate) fn output(&self, reference: StreamReference) -> &OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &self.mode.outputs[ix],
        }
    }

    /// Returns a mutable `Vec` with the output streams in the specification.
    pub(crate) fn outputs_as_mut<'a>(&'a mut self) -> &'a mut Vec<OutputStream> {
        &mut self.mode.outputs
    }

    /// Returns a `Vec` containing a reference for each stream in the specification.
    pub(crate) fn all_streams(&self) -> Vec<StreamReference> {
        self.input_refs()
            .iter()
            .map(|ix| StreamReference::InRef(*ix))
            .chain(self.output_refs().iter().map(|ix| StreamReference::OutRef(*ix)))
            .collect()
    }

    /// Returns a `Vec` containing a reference to each triggers in the specification.
    pub(crate) fn triggers(&self) -> Vec<&Trigger> {
        self.mode.triggers.iter().map(|t| t).collect()
    }

    /// Returns a mutable `Vec` with the trigger in the specification.
    pub(crate) fn triggers_as_mut<'a>(&'a mut self) -> &'a mut Vec<Trigger> {
        &mut self.mode.triggers
    }

    /// Returns a `Vec` containing a reference for each event-driven output stream in the specification.
    pub(crate) fn event_driven(&self) -> Vec<&EventDrivenStream> {
        self.mode.event_driven.iter().map(|e| e).collect()
    }

    /// Returns a mutable `Vec` with the event-driven output streams in the specification.
    pub(crate) fn event_driven_as_mut<'a>(&'a mut self) -> &'a mut Vec<EventDrivenStream> {
        &mut self.mode.event_driven
    }

    /// Returns a `Vec` containing a reference for each time-driven output stream in the specification.
    pub(crate) fn time_driven(&self) -> Vec<&TimeDrivenStream> {
        self.mode.time_driven.iter().map(|t| t).collect()
    }

    /// Returns a mutable `Vec` with the time-driven output streams in the specification.
    pub(crate) fn time_driven_as_mut<'a>(&'a mut self) -> &'a mut Vec<TimeDrivenStream> {
        &mut self.mode.time_driven
    }

    /// Returns a reference to a `Vec` containing a reference for each sliding in the specification.
    pub(crate) fn sliding_windows(&self) -> Vec<WindowReference> {
        (0..self.mode.sliding_windows.len()).map(WindowReference).collect::<Vec<WindowReference>>()
    }

    /// Returns a `Vec` containing a reference for each sliding window in the specification.
    pub(crate) fn sliding_window(&self, window: WindowReference) -> &SlidingWindow {
        &self.mode.sliding_windows[window.0]
    }

    /// Returns a mutable `Vec` with the sliding windows in the specification.
    pub(crate) fn sliding_windows_as_mut<'a>(&'a mut self) -> &'a mut Vec<SlidingWindow> {
        &mut self.mode.sliding_windows
    }

    // // /// Returns a `Vec` containing a reference to an output stream representing a trigger in the specification.
    // // pub(crate) fn get_triggers(&self) -> Vec<&OutputStream> {
    // //     self.triggers.iter().map(|t| self.get_out(t.reference)).collect()
    // // }

    // // /// Returns a `Vec` containing a reference for each event-driven output stream in the specification.
    // // pub(crate) fn get_event_driven(&self) -> Vec<&OutputStream> {
    // //     self.event_driven.iter().map(|t| self.get_out(t.reference)).collect()
    // // }

    // // /// Returns a `Vec` containing a reference for each time-driven output stream in the specification.
    // // pub(crate) fn get_time_driven(&self) -> Vec<&OutputStream> {
    // //     self.time_driven.iter().map(|t| self.get_out(t.reference)).collect()
    // // }

    // TODO we do not need these functions in the HIR
    // /// Provides a representation for the evaluation layers of all event-driven output streams.  Each element of the outer `Vec` represents a layer, each element of the inner `Vec` a stream in the layer.
    // pub(crate) fn get_event_driven_layers(&self) -> Vec<Vec<OutputReference>> {
    //     if self.event_driven.is_empty() {
    //         return vec![];
    //     }

    //     // Zip eval layer with stream reference.
    //     let streams_with_layers: Vec<(usize, OutputReference)> = self
    //         .event_driven
    //         .iter()
    //         .map(|s| s.reference)
    //         .map(|r| (self.get_out(r).eval_layer() as usize, r.out_ix()))
    //         .collect();

    //     // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
    //     // layers without entries and more layers than streams.
    //     // Minimization works as follows:
    //     // a) Find the greatest layer
    //     // b) For each potential layer...
    //     // c) Find streams that would be in it.
    //     // d) If there is none, skip this layer
    //     // e) If there are some, add them as layer.

    //     // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
    //     let max_layer = streams_with_layers.iter().max_by_key(|(layer, _)| layer).unwrap().0;

    //     let mut layers = Vec::new();
    //     // b) For each potential layer
    //     for i in 0..=max_layer {
    //         // c) Find streams that would be in it.
    //         let in_layer_i: Vec<OutputReference> =
    //             streams_with_layers.iter().filter_map(|(l, r)| if *l == i { Some(*r) } else { None }).collect();
    //         if in_layer_i.is_empty() {
    //             // d) If there is none, skip this layer
    //             continue;
    //         } else {
    //             // e) If there are some, add them as layer.
    //             layers.push(in_layer_i);
    //         }
    //     }
    //     layers
    // }

    // TODO we do not need these functions in the HIR
    // /// Computes a schedule for all time-driven streams.
    // pub(crate) fn compute_schedule(&self) -> Result<Schedule, String> {
    //     Schedule::from(self)
    // }
}

impl FullInformationHirMode {
    pub(crate) fn new() -> FullInformationHirMode {
        FullInformationHirMode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            time_driven: Vec::new(),
            event_driven: Vec::new(),
            sliding_windows: Vec::new(),
            triggers: Vec::new(),
        }
    }
}

impl Type {
    /// Indicates how many bytes a type requires to be stored in memory.
    pub(crate) fn size(&self) -> Option<ValSize> {
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

impl InputStream {
    pub(crate) fn get_name(&self) -> &String {
        &self.name
    }

    pub(crate) fn get_ty(&self) -> &Type {
        &self.ty
    }

    pub(crate) fn get_dependent_streams(&self) -> &Vec<Tracking> {
        &self.dependent_streams
    }

    pub(crate) fn get_dependent_windows(&self) -> &Vec<WindowReference> {
        &self.dependent_windows
    }

    pub(crate) fn get_layer(&self) -> u32 {
        self.layer
    }

    pub(crate) fn get_memory_bound(&self) -> MemorizationBound {
        self.memory_bound
    }

    pub(crate) fn get_reference(&self) -> StreamReference {
        self.reference
    }
}

impl OutputStream {
    pub(crate) fn get_name(&self) -> &String {
        &self.name
    }

    pub(crate) fn get_ty(&self) -> &Type {
        &self.ty
    }

    pub(crate) fn get_expr(&self) -> &Expression {
        &self.expr
    }

    pub(crate) fn get_input_dependencies(&self) -> &Vec<StreamReference> {
        &self.input_dependencies
    }

    pub(crate) fn get_outgoing_dependencies(&self) -> &Vec<Dependency> {
        &self.outgoing_dependencies
    }

    pub(crate) fn get_dependent_streams(&self) -> &Vec<Tracking> {
        &self.dependent_streams
    }

    pub(crate) fn get_dependent_windows(&self) -> &Vec<WindowReference> {
        &self.dependent_windows
    }

    pub(crate) fn get_layer(&self) -> u32 {
        self.layer
    }

    pub(crate) fn get_memory_bound(&self) -> MemorizationBound {
        self.memory_bound
    }

    pub(crate) fn get_reference(&self) -> StreamReference {
        self.reference
    }

    pub(crate) fn get_ac(&self) -> &Option<Activation<StreamReference>> {
        &self.ac
    }
}

impl SlidingWindow {
    pub(crate) fn get_target(&self) -> StreamReference {
        self.target
    }

    pub(crate) fn get_duration(&self) -> Duration {
        self.duration
    }

    pub(crate) fn get_wait(&self) -> bool {
        self.wait
    }

    pub(crate) fn get_op(&self) -> WindowOperation {
        self.op
    }

    pub(crate) fn get_reference(&self) -> WindowReference {
        self.reference
    }

    pub(crate) fn get_ty(&self) -> &Type {
        &self.ty
    }
}
