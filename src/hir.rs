#![allow(dead_code)]

/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occurs in different modes, adding different information to the intermediate representation.
*/
use crate::common_ir::StreamReference as SRef;
use crate::common_ir::*;
use crate::hir::expression::{DiscreteWindow, ExprId, SlidingWindow};
use crate::parse;

pub mod expression;
pub mod function_lookup;
pub(crate) mod lowering;
pub mod modes;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for MIR

use crate::reporting::Span;
use modes::HirMode;
use uom::si::rational64::Frequency as UOM_Frequency;

#[derive(Debug, Clone)]
pub struct RTLolaHIR<M: HirMode> {
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    triggers: Vec<Trigger>,
    next_input_ref: usize,
    next_output_ref: usize,
    mode: M,
}

pub type Hir<M> = RTLolaHIR<M>;

impl<M: HirMode> Hir<M> {
    pub fn inputs(&self) -> impl Iterator<Item = &Input> {
        self.inputs.iter()
    }

    pub fn outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter()
    }

    pub fn triggers(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.iter()
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub fn num_triggers(&self) -> usize {
        self.triggers.len()
    }

    pub fn all_streams<'a>(&'a self) -> impl Iterator<Item = SRef> + 'a {
        self.inputs
            .iter()
            .map(|i| i.sr)
            .chain(self.outputs.iter().map(|o| o.sr))
            .chain(self.triggers.iter().map(|t| t.sr))
    }
    pub fn get_input_with_name(&self, name: &str) -> Option<&Input> {
        self.inputs.iter().find(|&i| i.name == name)
    }
    pub fn get_output_with_name(&self, name: &str) -> Option<&Output> {
        self.outputs.iter().find(|&o| o.name == name)
    }
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the stream.
    pub name: String,
    /// The reference pointing to this stream.
    pub sr: SRef,
    /// The user annotated Type
    pub annotated_type: AnnotatedType,
    /// The code span the input represents
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Window {
    pub expr: ExprId,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Output {
    /// The name of the stream.
    pub name: String,
    /// The user annotated Type
    pub annotated_type: Option<AnnotatedType>,
    /// The activation condition, which defines when a new value of a stream is computed. In periodic streams, the condition is 'None'
    pub activation_condition: Option<AC>,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Parameter>,
    /// The declaration of the stream template for parametrized streams, e.g., the invoke declaration.
    pub instance_template: InstanceTemplate,
    /// The stream expression of a output stream, e.g., a + b.offset(by: -1).defaults(to: 0)
    pub expr_id: ExprId,
    /// The reference pointing to this stream.
    pub sr: StreamReference,
    /// The code span the output represents
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Parameter {
    /// The name of this parameter
    pub name: String,
    /// The annotated type of this parameter
    pub annotated_type: Option<AnnotatedType>,
    /// The id, index in the parameter vector in the output stream, for this parameter
    pub idx: usize,
    /// The code span of the parameter
    pub span: Span,
}

/// Use to hold either a frequency or an expression for the annotated activation condition
#[derive(Debug, Clone, PartialEq)]
pub enum AC {
    Frequency { span: Span, value: UOM_Frequency },
    Expr(ExprId),
}

#[derive(Debug, Clone, Copy)]
pub struct InstanceTemplate {
    /// The invoke condition of the parametrized stream.
    pub spawn: Option<SpawnTemplate>,
    /// The extend condition of the parametrized stream.
    pub filter: Option<ExprId>,
    /// The termination condition of the parametrized stream.
    pub close: Option<ExprId>,
}

#[derive(Debug, Clone, Copy)]
pub struct SpawnTemplate {
    /// The expression defining the parameter instances. If the stream has more than one parameter, the expression needs to return a tuple, with one element for each parameter
    pub target: ExprId,
    /// An additional condition for the creation of an instance, i.e., an instance is only created if the condition is true If 'is_true' is false, this component is assigned to 'None'
    pub condition: Option<ExprId>,
    /// A flag to describe if the invoke declaration contains an additional condition
    pub is_if: bool,
}

#[derive(Debug, Clone)]
pub struct Trigger {
    pub name: String,
    pub message: String,
    pub expr_id: ExprId,
    pub sr: SRef,
    /// The code span the trigger represents
    pub span: Span,
}

impl Trigger {
    fn new(name: Option<parse::Ident>, msg: Option<String>, expr_id: ExprId, sr: SRef, span: Span) -> Self {
        let name_str = name.map(|ident| ident.name).unwrap_or_else(String::new);
        Self { name: name_str, message: msg.unwrap_or_else(String::new), expr_id, sr, span }
    }
}

/// Represents the annotated given type for constants, input streams, etc.
/// It is converted from the AST type and an input for the typechecker.
/// After typechecking HirType is used to represent all type information.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AnnotatedType {
    //Can be annotated
    Int(u32),
    Float(u32),
    UInt(u32),
    Bool,
    String,
    Bytes,
    Option(Box<AnnotatedType>),
    Tuple(Vec<AnnotatedType>),
    //Used in function declaration
    Numeric,
    Param(usize, String),
}
