use std::fmt::Debug;
use std::time::Duration;

use itertools::Either;
use rtlola_parser::ast::WindowOperation;
use rtlola_reporting::Span;

use super::WindowReference;
use crate::hir::{AnnotatedType, Offset, SRef, StreamReference, WRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExprId(pub(crate) u32);

/// Represents an expression.
#[derive(Debug, Clone)]
pub struct Expression {
    /// The kind of expression.
    pub kind: ExpressionKind,

    pub(crate) eid: ExprId,

    pub(crate) span: Span,
}

impl Expression {
    pub fn id(&self) -> ExprId {
        self.eid
    }

    pub fn span(&self) -> Span {
        self.span.clone()
    }

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
    /// The Expression vector contains the arguments for a parametrized stream access
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
    Function(FnExprKind),
    Widen(WidenExprKind),
    /// Transforms an optional value into a "normal" one
    Default {
        /// The expression that results in an optional value.
        expr: Box<Expression>,
        /// An infallible expression providing a default value of `expr` evaluates to `None`.
        default: Box<Expression>,
    },
}

#[derive(Debug, Clone)]
pub struct FnExprKind {
    pub name: String,
    pub args: Vec<Expression>,
    pub(crate) type_param: Vec<AnnotatedType>,
}

#[derive(Debug, Clone)]
pub struct WidenExprKind {
    pub expr: Box<Expression>,
    pub(crate) ty: AnnotatedType,
}
/// Represents a constant value of a certain kind.
#[derive(Debug, Clone)]
pub enum Literal {
    Str(String),
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

#[derive(Debug, PartialEq, Clone, Eq)]
pub enum Constant {
    Basic(Literal),
    Inlined(Inlined),
}

#[derive(Debug, PartialEq, Clone, Eq)]
pub struct Inlined {
    pub lit: Literal,
    pub(crate) ty: AnnotatedType,
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

pub trait WindowAggregation: Debug + Copy {
    fn wait_until_full(&self) -> bool;
    fn operation(&self) -> WindowOperation;
    fn duration(&self) -> Either<Duration, usize>;
}

#[derive(Clone, Debug, Copy)]
pub struct SlidingAggr {
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
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

#[derive(Clone, Debug, Copy)]
pub struct DiscreteAggr {
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Window<Aggr: WindowAggregation> {
    /// The stream whose values will be aggregated.
    pub target: SRef,
    /// The stream calling and evaluating this window.
    pub caller: SRef,
    /// The duration over which the window aggregates.
    pub aggr: Aggr,
    /// A reference to this sliding window.
    pub(crate) reference: WRef,
    /// The Id of the expression in which this window is accessed. NOT the id of the window.
    pub(crate) eid: ExprId,
}

impl<A: WindowAggregation> Window<A> {
    pub fn reference(&self) -> WindowReference {
        self.reference
    }

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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StreamAccessKind {
    Sync,
    DiscreteWindow(WRef),
    SlidingWindow(WRef),
    Hold,
    Offset(Offset),
}
