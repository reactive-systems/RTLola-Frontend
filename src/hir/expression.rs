use std::time::Duration;

use super::WindowOperation;
use crate::hir::AnnotatedType;
use crate::{
    common_ir::StreamAccessKind, common_ir::StreamReference as SRef, common_ir::WindowReference as WRef,
    reporting::Span,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
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
/// Represents a constant value of a certain kind.
#[derive(Debug, Clone)]
pub enum ConstantLiteral {
    #[allow(missing_docs)]
    Str(String),
    #[allow(missing_docs)]
    Bool(bool),
    #[allow(missing_docs)]
    /// Integer constant with unknown sign
    Integer(i64),
    #[allow(missing_docs)]
    /// Integer constant known to be signed
    SInt(i128),
    #[allow(missing_docs)]
    /// Floating point constant
    Float(f64),
    //Frequency(UOM_Frequency),
    //Numeric(String, Option<String>),
}

#[derive(Debug, PartialEq, Clone, Eq)]
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

pub(crate) trait ValueEq {
    fn value_eq(&self, other: &Self) -> bool;
    fn value_neq(&self, other: &Self) -> bool {
        !self.value_eq(other)
    }
}

impl ValueEq for ExpressionKind {
    fn value_eq(&self, other: &Self) -> bool {
        use self::ExpressionKind::*;
        match (self, other) {
            (ParameterAccess(sref, idx), ParameterAccess(sref2, idx2)) => sref == sref2 && idx == idx2,
            (LoadConstant(c1), LoadConstant(c2)) => c1 == c2,
            (ArithLog(op, args), ArithLog(op2, args2)) => {
                op == op2 && args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2))
            }
            (StreamAccess(sref, kind, args), StreamAccess(sref2, kind2, args2)) => {
                sref == sref2 && kind == kind2 && args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2))
            }
            (
                Ite { condition: c1, consequence: c2, alternative: c3 },
                Ite { condition: b1, consequence: b2, alternative: b3 },
            ) => c1.value_eq(&b1) && c2.value_eq(&b2) && c3.value_eq(&b3),
            (Tuple(args), Tuple(args2)) => args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2)),
            (TupleAccess(inner, i1), TupleAccess(inner2, i2)) => i1 == i2 && inner.value_eq(&inner2),
            (Function { name, args, type_param }, Function { name: name2, args: args2, type_param: type_param2 }) => {
                name == name2
                    && type_param == type_param2
                    && args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2))
            }
            (Widen(inner, t1), Widen(inner2, t2)) => t1 == t2 && inner.value_eq(&inner2),
            (Default { expr, default }, Default { expr: expr2, default: default2 }) => {
                expr.value_eq(&expr2) && default.value_eq(&default2)
            }
            _ => false,
        }
    }
}

impl PartialEq for ConstantLiteral {
    fn eq(&self, other: &Self) -> bool {
        use self::ConstantLiteral::*;
        match (self, other) {
            (Float(f1), Float(f2)) => f64::abs(f1 - f2) < 0.00001f64,
            (Str(s1), Str(s2)) => s1 == s2,
            (Bool(b1), Bool(b2)) => b1 == b2,
            (Integer(i1), Integer(i2)) => i1 == i2,
            (SInt(i1), SInt(i2)) => i1 == i2,
            _ => false,
        }
    }
}

impl Eq for ConstantLiteral {}

impl ValueEq for Expression {
    fn value_eq(&self, other: &Self) -> bool {
        self.kind.value_eq(&other.kind)
    }
}
