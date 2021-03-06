use std::fmt::Debug;
use std::time::Duration;

use itertools::Either;
use rtlola_parser::ast::WindowOperation;
use rtlola_reporting::Span;

use super::WindowReference;
use crate::hir::{AnnotatedType, Offset, SRef, StreamReference, WRef};

/// Representation of the Id of an [Expression]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExprId(pub(crate) u32);

/// Representation of an expression in the [RtLolaHir](crate::hir::RtLolaHir).
///
/// An expression contains its kind, its id and its position in the specification.
#[derive(Debug, Clone)]
pub struct Expression {
    /// The kind of the expression
    pub kind: ExpressionKind,
    /// The [ExprId] of the expression
    pub(crate) eid: ExprId,
    /// The position of the expression in the specification
    pub(crate) span: Span,
}

impl Expression {
    /// Returns the [ExprId]] of the [Expression]
    pub fn id(&self) -> ExprId {
        self.eid
    }

    /// Returns the [Span] of the [Expression] identifying its position in the specification.
    pub fn span(&self) -> Span {
        self.span.clone()
    }

    /// Returns all streams that are synchronous accesses
    ///
    /// This function iterates over the [Expression] and retruns a vector of [StreamReference] identifying each stream that is synchronous accessed with its unique ID.
    pub(crate) fn get_sync_accesses(&self) -> Vec<StreamReference> {
        match &self.kind {
            ExpressionKind::ArithLog(_, children)
            | ExpressionKind::Tuple(children)
            | ExpressionKind::Function(FnExprKind { args: children, .. }) => {
                children.iter().flat_map(|c| c.get_sync_accesses()).collect()
            },
            ExpressionKind::StreamAccess(target, kind, children) => {
                match kind {
                    StreamAccessKind::Sync | StreamAccessKind::DiscreteWindow(_) => {
                        vec![*target]
                            .into_iter()
                            .chain(children.iter().flat_map(|c| c.get_sync_accesses()))
                            .collect()
                    },
                    _ => children.iter().flat_map(|c| c.get_sync_accesses()).collect(),
                }
            },
            ExpressionKind::Ite {
                condition,
                consequence,
                alternative,
            } => {
                condition
                    .as_ref()
                    .get_sync_accesses()
                    .into_iter()
                    .chain(consequence.as_ref().get_sync_accesses())
                    .chain(alternative.as_ref().get_sync_accesses())
                    .collect()
            },
            ExpressionKind::TupleAccess(child, _) | ExpressionKind::Widen(WidenExprKind { expr: child, .. }) => {
                child.as_ref().get_sync_accesses()
            },
            ExpressionKind::Default { expr, default } => {
                expr.as_ref()
                    .get_sync_accesses()
                    .into_iter()
                    .chain(default.as_ref().get_sync_accesses())
                    .collect()
            },
            _ => vec![],
        }
    }
}

impl ValueEq for Expression {
    fn value_eq(&self, other: &Self) -> bool {
        self.kind.value_eq(&other.kind)
    }
}

/// The kinds of an [Expression] of the [RtLolaHir](crate::hir::RtLolaHir).
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// Loading a [Constant]
    LoadConstant(Constant),
    /// Applying arithmetic or logic operation
    ///
    /// The first argument contains the operator of type [ArithLogOp], the second arguments contains the arguments of the operation, which are [Expressions](Expression). The vectors is structured as:
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp, Vec<Expression>),
    /// Accessing another stream
    ///
    /// A stream access has the following arguments:
    /// * the [StreamReference] of the stream that is accessed
    /// * the [StreamAccessKind] of the stream access, e.g. an offset access
    /// * the argmuents for parametrized stream accesses. This vector is empty if the stream that is accessed is not parametrized.
    StreamAccess(SRef, StreamAccessKind, Vec<Expression>),
    /// Accessing the n'th parameter of a parameterized stream
    ///
    /// This kind represents the access of a parameterized stream. For this, we use the folloing arguments:
    /// * the first argument contains the [StreamReference] of the parametrized stream that is accessed
    /// * the second argument contains the index of the parameter.
    ParameterAccess(SRef, usize),
    /// An if-then-else expression
    ///
    /// If the condition evaluates to true, the consequence is executed otherwise the alternative. All arguments are an [Expression].
    Ite {
        /// The condition of the if-then-else expression.
        condition: Box<Expression>,
        /// The consequence of the if-then-else expression.
        consequence: Box<Expression>,
        /// The alternative of the if-then-else expression.
        alternative: Box<Expression>,
    },
    /// A tuple expression.
    Tuple(Vec<Expression>),
    /// Represents an access to a tuple element
    ///
    /// The second argument indicates the index of the accessed element, while the first produces the accessed tuple.
    TupleAccess(Box<Expression>, usize),
    /// A function call with its monomorphic type
    Function(FnExprKind),
    /// A function call to widen the type of an [Expression]
    Widen(WidenExprKind),
    /// Represents the transformation of an optional value into a "normal" one
    Default {
        /// The expression that results in an optional value
        expr: Box<Expression>,
        /// An infallible expression providing a default value of `expr` evaluates to `None`
        default: Box<Expression>,
    },
}

/// Representation of an function call
//
/// The struction contains all information for a function call in the [ExpressionKind] enum.
#[derive(Debug, Clone)]
pub struct FnExprKind {
    /// The name of the function.
    pub name: String,
    /// The arguments of the function call.
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    pub args: Vec<Expression>,
    /// The type annoatation of the
    pub(crate) type_param: Vec<AnnotatedType>,
}

/// Representation of the function call to widen the type of an [Expression]
///
/// The struction contains all information to widen an [Expression] in the [ExpressionKind] enum.
#[derive(Debug, Clone)]
pub struct WidenExprKind {
    /// The [Expression] on which the function is called
    pub expr: Box<Expression>,
    /// The new type of `expr`
    pub(crate) ty: AnnotatedType,
}
/// Represents a constant value of a certain kind.
#[derive(Debug, Clone)]
pub enum Literal {
    /// String constant
    Str(String),
    /// Boolean constant
    Bool(bool),
    /// Integer constant with unknown sign
    Integer(i64),
    /// Integer constant known to be signed
    SInt(i128),
    /// Floating point constant
    Float(f64),
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        use self::Literal::*;
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

impl Eq for Literal {}

/// Represents a constant in the [ExpressionKind] enum of the [RtLolaHir](crate::hir::RtLolaHir).
///
/// The [RtLolaHir](crate::hir::RtLolaHir) differentiates between two types of constants:
/// * Constant expressions that are declared with a name and a [Type](rtlola_parser::ast::Type), which are inline in [crate::from_ast]
/// * Constant expressions occurring in an stream expression
///
/// Example:
/// constant a: Int8 := 5
/// output out := a    +    5
///               ^         ^
///               |         |
///            inlined    basic
#[derive(Debug, PartialEq, Clone, Eq)]
pub enum Constant {
    /// Basic constants occurring in stream expressions
    Basic(Literal),
    /// Inlined values of constant streams that are declared in the specification
    Inlined(Inlined),
}

/// Represents inlined constant values from constant streams
#[derive(Debug, PartialEq, Clone, Eq)]
pub struct Inlined {
    /// The value of the constant
    pub lit: Literal,
    /// The type of the constant
    pub(crate) ty: AnnotatedType,
}
/// Representation of the different stream accesses
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StreamAccessKind {
    /// Represents the synchronous access
    Sync,
    /// Represents the access to a (discrete window)[DiscreteAggr]
    ///
    /// The argument contains the reference to the (discrete window)[DiscreteAggr] whose value is used in the [Expression].
    DiscreteWindow(WRef),
    /// Represents the access to a (sliding window)[SlidingAggr]
    ///
    /// The argument contains the reference to the (sliding window)[SlidingAggr] whose value is used in the [Expression].
    SlidingWindow(WRef),
    /// Representation of sample and hold accesses
    Hold,
    /// Representation of offset accesses
    ///
    /// The argument contains the [Offset] of the stream access.
    Offset(Offset),
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

/// Functionality of [sliding window](SlidingAggr) and [discrete window](DiscreteAggr) aggregations
pub trait WindowAggregation: Debug + Copy {
    /// Returns wheter or not the first aggregated value will be produced immediately or wheter the window waits
    ///
    /// The function returns `true` if the windows waits until the [Duration] has passed at least once. Otherwise the function returns `false`.
    fn wait_until_full(&self) -> bool;
    /// Returns the [WindowOperation] of the sliding or discrete window
    fn operation(&self) -> WindowOperation;
    /// Returns the duration of the window
    ///
    /// The function returns the duration of a [sliding window](SlidingAggr) or the number of values used for a [discrete window](DiscreteAggr).
    fn duration(&self) -> Either<Duration, usize>;
}

/// Represents a sliding window aggregation
///
/// The struct contains all information that is specific for a sliding window aggregation. The data that is shared between a sliding window aggregation and a discrete window aggregation is stored a [Window].
#[derive(Clone, Debug, Copy, PartialEq)]
pub struct SlidingAggr {
    /// Flag to indicate whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation
    pub op: WindowOperation,
    /// The duration of the window
    ///
    /// The duration of a sliding window is a time span.
    pub duration: Duration,
}

impl WindowAggregation for SlidingAggr {
    fn wait_until_full(&self) -> bool {
        self.wait
    }

    fn operation(&self) -> WindowOperation {
        self.op
    }

    fn duration(&self) -> Either<Duration, usize> {
        Either::Left(self.duration)
    }
}
/// Represents a discrete window aggregation
///
/// The struct contains all information that is specific for a discrete window aggregation. The data that is shared between a sliding window aggregation and a discrete window aggregation is stored a [Window].

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct DiscreteAggr {
    /// Flag to indicate whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation
    pub op: WindowOperation,
    /// The duration of the window
    ///
    /// The duration of a discrete window is a discrete number of values.
    pub duration: usize,
}

impl WindowAggregation for DiscreteAggr {
    fn wait_until_full(&self) -> bool {
        self.wait
    }

    fn operation(&self) -> WindowOperation {
        self.op
    }

    fn duration(&self) -> Either<Duration, usize> {
        Either::Right(self.duration)
    }
}

/// Represents an instance of a sliding or a discrete window aggregation
///
/// The generatic `Aggr` defines if the instance is a slinding window or a discrete window.
/// The field `aggr` contains the data that is specific for a discrete of sliding window.
/// The other data is used for a discrete and a sliding window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Window<Aggr: WindowAggregation> {
    /// The stream whose values will be aggregated
    pub target: SRef,
    /// The stream calling and evaluating this window
    pub caller: SRef,
    /// The data that differentiates a sliding and a discrete window
    ///
    /// This field can either has the type [SlidingAggr] or [DiscreteAggr].
    pub aggr: Aggr,
    /// The reference of this window.
    pub(crate) reference: WRef,
    /// The Id of the expression in which this window is accessed
    ///
    /// This field contains the Id of the expression that uses the produced value. It is NOT the id of the window.
    pub(crate) eid: ExprId,
}

impl<A: WindowAggregation> Window<A> {
    /// Returns the reference of the window
    pub fn reference(&self) -> WindowReference {
        self.reference
    }

    /// Returns the Id of the expression in which this window is accessed
    ///
    /// The return value contains the Id of the expression that uses the produced value. This value is NOT the id of the window.
    pub fn id(&self) -> ExprId {
        self.eid
    }
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
            },
            (StreamAccess(sref, kind, args), StreamAccess(sref2, kind2, args2)) => {
                sref == sref2 && kind == kind2 && args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2))
            },
            (
                Ite {
                    condition: c1,
                    consequence: c2,
                    alternative: c3,
                },
                Ite {
                    condition: b1,
                    consequence: b2,
                    alternative: b3,
                },
            ) => c1.value_eq(&b1) && c2.value_eq(&b2) && c3.value_eq(&b3),
            (Tuple(args), Tuple(args2)) => args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2)),
            (TupleAccess(inner, i1), TupleAccess(inner2, i2)) => i1 == i2 && inner.value_eq(&inner2),
            (
                Function(FnExprKind { name, args, type_param }),
                Function(FnExprKind {
                    name: name2,
                    args: args2,
                    type_param: type_param2,
                }),
            ) => {
                name == name2
                    && type_param == type_param2
                    && args.iter().zip(args2.iter()).all(|(a1, a2)| a1.value_eq(&a2))
            },
            (Widen(WidenExprKind { expr: inner, ty: t1 }), Widen(WidenExprKind { expr: inner2, ty: t2 })) => {
                t1 == t2 && inner.value_eq(&inner2)
            },
            (
                Default { expr, default },
                Default {
                    expr: expr2,
                    default: default2,
                },
            ) => expr.value_eq(&expr2) && default.value_eq(&default2),
            _ => false,
        }
    }
}
