use std::time::Duration;

use super::WindowOperation;
use crate::hir::AnnotatedType;
use crate::{common_ir::Offset, common_ir::StreamReference as SRef, common_ir::WindowReference as WRef, parse::Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

/// Represents an expression.
#[derive(Debug, Clone)]
pub struct Expression {
    /// The kind of expression.
    pub kind: ExpressionKind,

    pub eid: ExprId,

    pub span: Span,
}

/// The expressions of the IR.
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// Loading a constant
    LoadConstant(Constant),
    /// Applying arithmetic or logic operation and its monomorphic type
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp, Vec<Expression>),
    /// Accessing another stream
    /// The Expression vector containsthe arguments for a parametrized stream access
    StreamAccess(SRef, StreamAccessKind, Vec<Expression>),
    /// Accessing the n'th parameter of this parameterized stream
    ParameterAccess(SRef, usize),
    /// An if-then-else expression
    Ite {
        condition: Box<Expression>,
        consequence: Box<Expression>,
        alternative: Box<Expression>,
    },
    /// A tuple expression
    Tuple(Vec<Expression>),
    /// Represents an access to a specific tuple element.  The second argument indicates the index of the accessed element while the first produces the accessed tuple.
    TupleAccess(Box<Expression>, usize),
    /// A function call with its monomorphic type
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    //Function(String, Vec<Expression>),
    Function {
        name: String,
        args: Vec<Expression>,
        type_param: Vec<AnnotatedType>,
    },
    Widen(Box<Expression>, AnnotatedType),
    /// Transforms an optional value into a "normal" one
    Default {
        /// The expression that results in an optional value.
        expr: Box<Expression>,
        /// An infallible expression providing a default value of `expr` evaluates to `None`.
        default: Box<Expression>,
    },
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StreamAccessKind {
    Sync,
    DiscreteWindow(WRef),
    SlidingWindow(WRef),
    Hold,
    Offset(Offset),
}

/// Represents a constant value of a certain kind.
#[derive(Debug, PartialEq, Clone)]
pub enum ConstantLiteral {
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
    Numeric(String, Option<String>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Constant {
    BasicConstant(ConstantLiteral),
    InlinedConstant(ConstantLiteral, AnnotatedType),
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

/// Represents an instance of a sliding window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SlidingWindow {
    /// The stream whose values will be aggregated.
    pub target: SRef,
    /// The stream calling and evaluating this window.
    pub caller: SRef,
    /// The duration over which the window aggregates.
    pub duration: Duration,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
    /// A reference to this sliding window.
    pub reference: WRef,
    /// The ExprId references the window location
    pub eid: ExprId,
}

/// Represents an instance of a discrete window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct DiscreteWindow {
    /// The stream whose values will be aggregated.
    pub target: SRef,
    /// The stream calling and evaluating this window.
    pub caller: SRef,
    /// The number of values over which the window aggregates.
    pub duration: u32,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
    /// A reference to this sliding window.
    pub reference: WRef,
    /// The ExprId references the window location
    pub eid: ExprId,
}
