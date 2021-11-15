use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::time::Duration;

use itertools::{iproduct, Either};
use rtlola_parser::ast::WindowOperation;
use rtlola_reporting::Span;

use super::WindowReference;
use crate::hir::{AnnotatedType, Hir, Offset, SRef, StreamReference, WRef};
use crate::modes::HirMode;

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
    fn value_eq(&self, other: &Self, parameter_map: &ExpressionContext) -> bool {
        self.kind.value_eq(&other.kind, parameter_map)
    }

    fn value_eq_ignore_parameters(&self, other: &Self) -> bool {
        self.kind.value_eq_ignore_parameters(&other.kind)
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
#[derive(Debug, PartialEq, Clone, Copy, Hash, Eq)]
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

/// A context for expressions that establishes equality between parameters of different streams based on their spawn expression.
// Maps a stream 'a' and a parameter 'x' of stream 'b' to the set of matching parameters of 'a'
#[derive(Debug, Clone)]
pub(crate) struct ExpressionContext(HashMap<SRef, HashMap<(SRef, usize), HashSet<usize>>>);

impl ExpressionContext {
    // Two parameters of two streams are equal if they are spawned with the same expression under the same condition.
    // For example:
    //
    // output a(x, y) spawn with (5, 42) if i = 2 ...
    // output b(v, w) spawn with (42, 7) if i = 2 ...
    //
    // Then parameter y of a is equal to the parameter v of b
    // The spawn condition has to match as otherwise y could already be initialized while v is not.
    //
    // This equivalence is computed as follows:
    //
    // For all pairs of output streams that have parameters and the same spawn condition:
    //      Get the vector of expressions that initialize the parameters of a (i.e. (5, 42))
    //      Get the vector of expressions that initialize the parameters of b (i.e. (42, 7))
    //      For all paris of expressions in the cross product of these two vectors (i.e. (5, 42), (5, 7), (42, 42), (42, 7)):
    //          check if the two expressions are equal
    //              if true then insert the corresponding parameter into the resulting set
    pub(crate) fn new<M: HirMode>(hir: &Hir<M>) -> ExpressionContext {
        let mut inner = HashMap::with_capacity(hir.outputs.len());
        for current in hir.outputs() {
            let mut para_mapping: HashMap<(SRef, usize), HashSet<usize>> = HashMap::new();

            let cur_spawn_cond = current.instance_template.spawn_condition(hir);

            let current_spawn_args = current.instance_template.spawn_arguments(hir);

            assert_eq!(current.params.len(), current_spawn_args.len());

            if !current.params.is_empty() {
                for target in hir.outputs() {
                    // if both have a spawn condition they must match
                    let target_spawn_cond = target.instance_template.spawn_condition(hir);
                    let cond_match = match (cur_spawn_cond, target_spawn_cond) {
                        (Some(e1), Some(e2)) => e1.value_eq_ignore_parameters(e2),
                        (None, None) => true,
                        _ => false,
                    };
                    if !target.params.is_empty() && cond_match {
                        let target_spawn_args = target.instance_template.spawn_arguments(hir);

                        assert_eq!(target.params.len(), target_spawn_args.len());

                        iproduct!(
                            current_spawn_args.iter().enumerate(),
                            target_spawn_args.iter().enumerate()
                        )
                        .filter_map(|((current_para, current_exp), (target_para, target_exp))| {
                            if current_exp.value_eq_ignore_parameters(target_exp) {
                                Some(((target.sr, target_para), current_para))
                            } else {
                                None
                            }
                        })
                        .for_each(|(k, v)| {
                            if let Some(paras) = para_mapping.get_mut(&k) {
                                paras.insert(v);
                            } else {
                                para_mapping.insert(k, vec![v].into_iter().collect::<HashSet<usize>>());
                            }
                        });
                    }
                }
            }

            inner.insert(current.sr, para_mapping);
        }
        ExpressionContext(inner)
    }

    /// Checks if the parameter of source matches the parameter of target
    pub(crate) fn matches(&self, source: SRef, source_parameter: usize, target: SRef, target_parameter: usize) -> bool {
        self.0
            .get(&source)
            .and_then(|para_map| para_map.get(&(target, target_parameter)))
            .map(|para_set| para_set.contains(&source_parameter))
            .unwrap_or(false)
    }

    #[cfg(test)]
    /// Extracts the parameter mapping for a single stream from the context
    /// Query the map for a stream b with a parameter p to get the parameter q of the stream if it matches with parameter p
    pub(crate) fn map_for(&self, stream: SRef) -> &HashMap<(SRef, usize), HashSet<usize>> {
        self.0
            .get(&stream)
            .expect("Invalid initialization of ExpressionContext")
    }
}

pub(crate) trait ValueEq {
    fn value_eq(&self, other: &Self, parameter_map: &ExpressionContext) -> bool;
    fn value_neq(&self, other: &Self, parameter_map: &ExpressionContext) -> bool {
        !self.value_eq(other, parameter_map)
    }

    fn value_eq_ignore_parameters(&self, other: &Self) -> bool;
    fn value_neq_ignore_parameters(&self, other: &Self) -> bool {
        !self.value_eq_ignore_parameters(other)
    }
}

impl ValueEq for ExpressionKind {
    fn value_eq(&self, other: &Self, parameter_map: &ExpressionContext) -> bool {
        use self::ExpressionKind::*;
        match (self, other) {
            (ParameterAccess(sref, idx), ParameterAccess(sref2, idx2)) => {
                parameter_map.matches(*sref, *idx, *sref2, *idx2)
            },
            (LoadConstant(c1), LoadConstant(c2)) => c1 == c2,
            (ArithLog(op, args), ArithLog(op2, args2)) => {
                op == op2
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq(a2, parameter_map))
            },
            (StreamAccess(sref, kind, args), StreamAccess(sref2, kind2, args2)) => {
                sref == sref2
                    && kind == kind2
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq(a2, parameter_map))
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
            ) => c1.value_eq(b1, parameter_map) && c2.value_eq(b2, parameter_map) && c3.value_eq(b3, parameter_map),
            (Tuple(args), Tuple(args2)) => {
                args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq(a2, parameter_map))
            },
            (TupleAccess(inner, i1), TupleAccess(inner2, i2)) => i1 == i2 && inner.value_eq(inner2, parameter_map),
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
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq(a2, parameter_map))
            },
            (Widen(WidenExprKind { expr: inner, ty: t1 }), Widen(WidenExprKind { expr: inner2, ty: t2 })) => {
                t1 == t2 && inner.value_eq(inner2, parameter_map)
            },
            (
                Default { expr, default },
                Default {
                    expr: expr2,
                    default: default2,
                },
            ) => expr.value_eq(expr2, parameter_map) && default.value_eq(default2, parameter_map),
            _ => false,
        }
    }

    fn value_eq_ignore_parameters(&self, other: &Self) -> bool {
        use ExpressionKind::*;
        match (self, other) {
            (ParameterAccess(sref, idx), ParameterAccess(sref2, idx2)) => sref == sref2 && idx == idx2,
            (LoadConstant(c1), LoadConstant(c2)) => c1 == c2,
            (ArithLog(op, args), ArithLog(op2, args2)) => {
                op == op2
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq_ignore_parameters(a2))
            },
            (StreamAccess(sref, kind, args), StreamAccess(sref2, kind2, args2)) => {
                sref == sref2
                    && kind == kind2
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq_ignore_parameters(a2))
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
            ) => {
                c1.value_eq_ignore_parameters(b1)
                    && c2.value_eq_ignore_parameters(b2)
                    && c3.value_eq_ignore_parameters(b3)
            },
            (Tuple(args), Tuple(args2)) => {
                args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq_ignore_parameters(a2))
            },
            (TupleAccess(inner, i1), TupleAccess(inner2, i2)) => i1 == i2 && inner.value_eq_ignore_parameters(inner2),
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
                    && args.len() == args2.len()
                    && args
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| a1.value_eq_ignore_parameters(a2))
            },
            (Widen(WidenExprKind { expr: inner, ty: t1 }), Widen(WidenExprKind { expr: inner2, ty: t2 })) => {
                t1 == t2 && inner.value_eq_ignore_parameters(inner2)
            },
            (
                Default { expr, default },
                Default {
                    expr: expr2,
                    default: default2,
                },
            ) => expr.value_eq_ignore_parameters(expr2) && default.value_eq_ignore_parameters(default2),
            _ => false,
        }
    }
}
