//! This module contains the [RtLolaAst] data structures for the RTLola Language.
//!
//! Every node in the abstract syntax tree is assigned a unique id and has a span referencing the node's location in the specification.

use std::cell::RefCell;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

mod conversion;
mod print;
use std::rc::Rc;

use num::rational::Rational64 as Rational;
use rtlola_reporting::Span;
/// The root of a RTLola specification, consisting of stream and trigger declarations.
/// Each declaration contains the id of the Ast node, a span, and declaration-specific components.
///
/// # Ast Node Kinds
/// * [Import] represents an import statement for a module.
/// * [Constant] represents a constant stream.
/// * [Input] represents an input stream.
/// * [Output] represents an output stream.
/// * [Mirror] represents mirror streams, a syntactic sugar for an output stream.
/// * [Trigger] represents a trigger declaration.
/// * [TypeDeclaration] captures a user given type declaration.
///
/// # Related Data Structures
/// * A [NodeId] is a unique identifier given to every node of the [RtLolaAst]
/// * A [Span] links an Ast node to its code location.
#[derive(Debug, Default, Clone)]
pub struct RtLolaAst {
    /// The imports of additional modules
    pub imports: Vec<Import>,
    /// The constant stream declarations
    pub constants: Vec<Rc<Constant>>,
    /// The input stream declarations
    pub inputs: Vec<Rc<Input>>,
    /// The output stream declarations
    pub outputs: Vec<Rc<Output>>,
    /// The mirror stream declarations
    pub mirrors: Vec<Rc<Mirror>>,
    /// The trigger declarations
    pub trigger: Vec<Rc<Trigger>>,
    /// The user-defined type declarations
    pub type_declarations: Vec<TypeDeclaration>,
    /// Next highest NodeId
    pub next_node_id: RefCell<NodeId>,
}

impl RtLolaAst {
    /// Creates a new and empty [RtLolaAst]
    pub(crate) fn empty() -> RtLolaAst {
        RtLolaAst {
            imports: Vec::new(),
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            mirrors: Vec::new(),
            trigger: Vec::new(),
            type_declarations: Vec::new(),
            next_node_id: RefCell::new(NodeId::default()),
        }
    }

    pub(crate) fn next_id(&self) -> NodeId {
        let res = *self.next_node_id.borrow();
        self.next_node_id.borrow_mut().id += 1;
        self.next_node_id.borrow_mut().prime_counter = 0;
        res
    }
}

/// An Ast node representing the import of a module, which brings additional implemented functionality to a specification.
/// The 'math' module, for example, adds pre-defined mathematical functions as the sine or cosine function.
#[derive(Debug, Clone)]
pub struct Import {
    /// The name of the module
    pub name: Ident,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the module
    pub span: Span,
}

/// An Ast node representing the declaration of a constant.
#[derive(Debug, Clone)]
pub struct Constant {
    /// The name of the constant stream
    pub name: Ident,
    /// The value type of the constant stream
    pub ty: Option<Type>,
    /// The literal defining the constant
    pub literal: Literal,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the constant stream
    pub span: Span,
}

/// An Ast node representing the declaration of an input stream.
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the input stream
    pub name: Ident,
    ///  The value type of the input stream
    pub ty: Type,
    /// The parameters of a parameterized input stream; The vector is empty in non-parametrized streams.
    pub params: Vec<Rc<Parameter>>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the input stream
    pub span: Span,
}

/// An Ast node representing the declaration of an output stream.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Output {
    /// The name of the output stream
    pub name: Ident,
    /// An optional value type annotation of the output stream
    pub annotated_type: Option<Type>,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Rc<Parameter>>,
    /// The spawn declaration of a parameterized stream
    pub spawn: Option<SpawnSpec>,
    /// The eval declaration of a stream,
    pub eval: Vec<EvalSpec>,
    ///  The close declaration of parametrized stream
    pub close: Option<CloseSpec>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the output stream
    pub span: Span,
}

/// Represents an output stream that mirrors another but filters them.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Mirror {
    /// The name of the mirror stream.
    pub name: Ident,
    /// The condition under which values of the target will be propagated.
    pub filter: Expression,
    /// The stream that is supposed to be mirrored.
    pub target: Ident,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the parameter
    pub span: Span,
}

/// An Ast node representing the declaration of a parameter of a parametrized stream.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Parameter {
    /// The name of the parameter
    pub name: Ident,
    /// An optional value type annotation of the parameter
    pub ty: Option<Type>,
    /// The index of this parameter in the list of parameter of the respective output stream
    pub param_idx: usize,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the parameter
    pub span: Span,
}

/// An Ast node representing the declaration of a spawn condition and expression of a stream.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SpawnSpec {
    /// The expression defining the parameter instances. If the stream has more than one parameter, the expression needs to return a tuple, with one element for each parameter
    pub expression: Option<Expression>,
    /// The pacing type describing when a new instance is created.
    pub annotated_pacing: Option<Expression>,
    /// An additional condition for the creation of an instance, i.e., an instance is only created if the condition is true.
    pub condition: Option<Expression>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the invoke declaration
    pub span: Span,
}

/// An Ast node representing the evaluation condition and expression of a stream
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct EvalSpec {
    /// The pacing type describing when a new value is computed.
    pub annotated_pacing: Option<Expression>,
    /// The boolean expression defining the condition, if a stream instance is evaluated.
    pub condition: Option<Expression>,
    /// The evaluated expression, defining the value of the stream.
    pub eval_expression: Option<Expression>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/// An Ast node representing the declaration of a close condition of a stream
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CloseSpec {
    /// The boolean expression defining the condition, if a stream instance is closed.
    pub condition: Expression,
    /// The pacing type describing when the close condition is evaluated.
    pub annotated_pacing: Option<Expression>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/// An Ast node representing the declaration of a trigger
#[derive(Debug, Clone)]
pub struct Trigger {
    /// The boolean expression of a trigger
    pub expression: Expression,
    /// The pacing type, which defines when a new value of a stream is computed.
    pub annotated_pacing_type: Option<Expression>,
    /// The optional trigger message, which is printed if the monitor raises the trigger
    pub message: Option<String>,
    /// A collection of streams which can be used in the message. Their value is printed when the trigger is fired.
    pub info_streams: Vec<Ident>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/// An Ast node representing the declaration of a user-defined type.
#[allow(clippy::vec_box)]
#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    /// The name of the new type.
    pub name: Option<Ident>,
    /// The components of the new type, e.g. a GPS type might consist of a type for the latitude and for the longitude
    pub fields: Vec<Box<TypeDeclField>>,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the type declaration
    pub span: Span,
}

/// An Ast node representing the declaration of a field of a user-defined type.
#[derive(Debug, Clone)]
pub struct TypeDeclField {
    /// The type of a field of a user-defined type
    pub ty: Type,
    /// The name of a field of a user-defined type
    pub name: String,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the type declaration
    pub span: Span,
}

/// An Ast node representing an opening or closing parenthesis.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Parenthesis {
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Parenthesis {
    /// Creates a new Parenthesis
    pub(crate) fn new(id: NodeId, span: Span) -> Parenthesis {
        Parenthesis { id, span }
    }
}

/// An Ast node representing the declaration of a value type
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Type {
    /// The kind of the type, e.g., a tuple
    pub kind: TypeKind,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Type {
    /// Creates a new non-recursive type like `Int` or `Bool`
    pub(crate) fn new_simple(id: NodeId, name: String, span: Span) -> Type {
        Type {
            id,
            kind: TypeKind::Simple(name),
            span,
        }
    }

    /// Creates a new tuple type
    pub(crate) fn new_tuple(id: NodeId, tuple: Vec<Type>, span: Span) -> Type {
        Type {
            id,
            kind: TypeKind::Tuple(tuple),
            span,
        }
    }

    /// Creates a new optional type
    pub(crate) fn new_optional(id: NodeId, name: Type, span: Span) -> Type {
        Type {
            id,
            kind: TypeKind::Optional(name.into()),
            span,
        }
    }
}

/// Ast representation of the value type of a stream
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum TypeKind {
    /// A simple type, e.g., `Int`
    Simple(String),
    /// A tuple type, e.g., `(Int32, Float32)`
    Tuple(Vec<Type>),
    /// An optional type, e.g., `Int?`
    Optional(Box<Type>),
}

/// The Ast representation of a stream expression
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Expression {
    /// The kind of the root expression, e.g., stream access
    pub kind: ExpressionKind,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Expression {
    /// Creates a new expression
    pub(crate) fn new(id: NodeId, kind: ExpressionKind, span: Span) -> Expression {
        Expression { kind, id, span }
    }
}

#[allow(clippy::large_enum_variant, clippy::vec_box)]
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
/// The Ast representation of a single expression
pub enum ExpressionKind {
    /// A literal, e.g., `1`, `"foo"`
    Lit(Literal),
    /// An identifier, e.g., `foo`
    Ident(Ident),
    /// Accessing a stream
    StreamAccess(Box<Expression>, StreamAccessKind),
    /// A default expression, e.g., `a.defaults(to: 0) `
    Default(Box<Expression>, Box<Expression>),
    /// An offset expression, e.g., `a.offset(by: -1)`
    Offset(Box<Expression>, Offset),
    /// A discrete window with a duration `duration` as an integer constant and aggregation function `aggregation`
    DiscreteWindowAggregation {
        /// The accesses stream
        expr: Box<Expression>,
        /// The duration of the window
        duration: Box<Expression>,
        /// Flag to mark that the window returns only a value if the complete duration has passed
        wait: bool,
        /// The aggregation function
        aggregation: WindowOperation,
    },
    /// A sliding window with duration `duration` and aggregation function `aggregation`
    SlidingWindowAggregation {
        /// The accesses stream
        expr: Box<Expression>,
        /// The duration of the window
        duration: Box<Expression>,
        /// Flag to mark that the window returns only a value if the complete duration has passed
        wait: bool,
        /// The aggregation function
        aggregation: WindowOperation,
    },
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, Box<Expression>, Box<Expression>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, Box<Expression>),
    /// An if-then-else expression
    Ite(Box<Expression>, Box<Expression>, Box<Expression>),
    /// An expression enveloped in parentheses
    ParenthesizedExpression(Option<Box<Parenthesis>>, Box<Expression>, Option<Box<Parenthesis>>),
    /// An expression was expected, e.g., after an operator like `*`
    MissingExpression,
    /// A tuple expression
    Tuple(Vec<Expression>),
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct field
    Field(Box<Expression>, Ident),
    /// A method call, e.g., `foo.bar(-1)`
    Method(Box<Expression>, FunctionName, Vec<Type>, Vec<Expression>),
    /// A function call
    Function(FunctionName, Vec<Type>, Vec<Expression>),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
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

/// Describes the operation used to access a stream
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum StreamAccessKind {
    /// Synchronous access
    Sync,
    /// Hold access for *incompatible* stream types, returns previous known value
    Hold,
    /// Optional access, returns value if it exists, called by `.get()`
    Get,
    /// Boolean Typed access, returning true if the target stream received a new value at the current timestamp. Called with `.is_fresh()`.
    Fresh,
}

/// Describes the operation used to access a stream with a offset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Offset {
    /// Discrete offset
    Discrete(i16),
    /// Real-time offset
    RealTime(Rational, TimeUnit),
}

/// Supported time unit for real time expressions
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    /// Note: A year is always, *always*, 365 days long.
    Year,
}

/// An Ast node representing the declaration of a literal
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Literal {
    /// The kind of the literal, e.g., boolean, string, numeric, ...
    pub kind: LitKind,
    /// The id of the node in the Ast
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Literal {
    /// Creates a new bool literal
    pub(crate) fn new_bool(id: NodeId, val: bool, span: Span) -> Literal {
        Literal {
            id,
            kind: LitKind::Bool(val),
            span,
        }
    }

    /// Creates a new numeric literal
    pub(crate) fn new_numeric(id: NodeId, val: &str, unit: Option<String>, span: Span) -> Literal {
        Literal {
            id,
            kind: LitKind::Numeric(val.to_string(), unit),
            span,
        }
    }

    /// Creates a new string literal
    pub(crate) fn new_str(id: NodeId, val: &str, span: Span) -> Literal {
        Literal {
            id,
            kind: LitKind::Str(val.to_string()),
            span,
        }
    }

    /// Creates a new raw string literal
    pub(crate) fn new_raw_str(id: NodeId, val: &str, span: Span) -> Literal {
        Literal {
            id,
            kind: LitKind::RawStr(val.to_string()),
            span,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// The Ast representation of literals
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(String),
    /// A raw string literal (`r#" x " a \ff "#`)
    RawStr(String),
    /// A numeric value with optional postfix part (`42`, `1.3`, `1Hz`, `100sec`)
    /// Stores as a string to have lossless representation
    Numeric(String, Option<String>),
    /// A boolean literal (`true`)
    Bool(bool),
}

/// An Ast node representing a binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
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

/// An Ast node representing an unary operator.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
    /// The `~` operator for one's complement
    BitNot,
}

/// An Ast node representing the name of a called function and also the names of the arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct FunctionName {
    /// The name of the function
    pub name: Ident,
    /// A list containing the name of each argument.  If the argument is unnamed, it is represented by `None`.
    pub arg_names: Vec<Option<Ident>>,
}

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
/// This struct represents an identifier in the specification.
/// For example the name of an [Output] or [Input].
pub struct Ident {
    /// The name of the identifier
    pub name: String,
    /// The span in the specification declaring the identifier
    pub span: Span,
}

impl Ident {
    /// Creates a new identifier.
    pub(crate) fn new(name: String, span: Span) -> Ident {
        Ident { name, span }
    }
}

/// In the equality definition of `Ident`, we only compare the string values
/// and ignore the `Span` info
impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Hash for Ident {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Every node in the Ast gets a unique id, represented by a 32bit unsigned integer.
/// They are used in the later analysis phases to store information about Ast nodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId {
    /// The actual unique id.
    pub id: u32,
    /// Counter to track transformations on this node. Increased during syntactic sugar removal.
    pub prime_counter: u32,
}

impl NodeId {
    /// Creates a new NodeId
    pub fn new(x: usize) -> NodeId {
        assert!(x < (u32::max_value() as usize));
        NodeId {
            id: x as u32,
            prime_counter: 0u32,
        }
    }

    /// Creates a copy NodeId with incremented prime counter, which indicates a applied transformation for desugarization.
    pub fn primed(&self) -> Self {
        let NodeId { id, prime_counter } = *self;
        NodeId {
            id,
            prime_counter: prime_counter + 1,
        }
    }
}
