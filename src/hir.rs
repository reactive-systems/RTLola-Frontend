#![allow(dead_code)]

/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occurs in different modes, adding different information to the intermediate representation.
*/
use crate::common_ir::StreamReference as SRef;
use crate::common_ir::*;
use crate::hir::expression::{ExprId, SlidingWindow};
use crate::parse;

pub(crate) mod expression;
pub(crate) mod function_lookup;
pub(crate) mod lowering;
pub(crate) mod modes;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for MIR

use modes::HirMode;

#[derive(Debug, Clone)]
pub(crate) struct RTLolaHIR<M: HirMode> {
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    triggers: Vec<Trigger>,
    next_input_ref: usize,
    next_output_ref: usize,
    mode: M,
}

pub(crate) type Hir<M> = RTLolaHIR<M>;

impl<M: HirMode> Hir<M> {
    pub(crate) fn inputs(&self) -> impl Iterator<Item = &Input> {
        self.inputs.iter()
    }

    pub(crate) fn outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter()
    }

    pub(crate) fn triggers(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.iter()
    }

    pub(crate) fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub(crate) fn num_triggers(&self) -> usize {
        self.triggers.len()
    }

    pub(crate) fn all_streams<'a>(&'a self) -> impl Iterator<Item = SRef> + 'a {
        self.inputs
            .iter()
            .map(|i| i.sr)
            .chain(self.outputs.iter().map(|o| o.sr))
            .chain(self.triggers.iter().map(|t| t.sr))
    }
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the stream.
    pub(crate) name: String,
    /// The reference pointing to this stream.
    pub(crate) sr: SRef,
    /// The user annotated Type
    pub annotated_type: AnnotatedType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Window {
    pub(crate) expr: ExprId,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, Clone)]
pub struct Output {
    /// The name of the stream.
    pub name: String,
    /// The user annotated Type
    pub annotated_type: Option<AnnotatedType>,
    /// The activation condition, which defines when a new value of a stream is computed. In periodic streams, the condition is 'None'
    pub activation_condition: Option<ExprId>,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Parameter>,
    /// The declaration of the stream template for parametrized streams, e.g., the invoke declaration.
    pub template_spec: InstanceTemplate,
    /// The stream expression of a output stream, e.g., a + b.offset(by: -1).defaults(to: 0)
    pub expr_id: ExprId,
    /// The reference pointing to this stream.
    pub sr: StreamReference,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Parameter {
    /// The name of this parameter
    pub name: String,
    /// The annotated type of this parameter
    pub annotated_type: Option<AnnotatedType>,
    /// The id, index in the parameter vector in the output stream, for this parameter
    pub idx: usize,
}

#[derive(Debug, Clone)]
pub struct InstanceTemplate {
    /// The invoke condition of the parametrized stream.
    pub spawn: Option<SpawnTemplate>,
    /// The extend condition of the parametrized stream.
    pub filter: Option<ExprId>,
    /// The termination condition of the parametrized stream.
    pub close: Option<ExprId>,
}

#[derive(Debug, Clone)]
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
    pub(crate) name: String,
    pub(crate) message: String,
    pub expr_id: ExprId,
    pub(crate) sr: SRef,
}

impl Trigger {
    fn new(name: Option<parse::Ident>, msg: Option<String>, expr_id: ExprId, sr: SRef) -> Self {
        let name_str = name.map(|ident| ident.name).unwrap_or_else(String::new);
        Self { name: name_str, message: msg.unwrap_or_else(String::new), expr_id, sr }
    }
}

#[derive(Debug, PartialEq, Clone)]
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
