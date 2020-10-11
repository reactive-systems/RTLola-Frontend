#![allow(dead_code)]

/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occurs in different modes, adding different information to the intermediate representation.
*/
use crate::common_ir::StreamReference as SRef;
use crate::common_ir::*;
use crate::hir::expression::ExprId;
use crate::parse;

pub(crate) mod expression;
pub(crate) mod lowering;
pub(crate) mod modes;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for MIR

use modes::HirMode;

#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct Input {
    /// The name of the stream.
    pub(crate) name: String,
    /// The reference pointing to this stream.
    pub(crate) sr: SRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Window {
    pub(crate) expr: ExprId,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Output {
    /// The name of the stream.
    pub(crate) name: String,
    /// The reference pointing to this stream.
    pub(crate) sr: StreamReference,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Trigger {
    pub(crate) name: String,
    pub(crate) message: String,
    pub(crate) sr: SRef,
}

impl Trigger {
    fn new(_name: Option<parse::Ident>, _msg: Option<String>, _sr: SRef) -> Self {
        let _name = unimplemented!();
        // Self { name, message: msg.unwrap_or_else(String::new), sr }
    }
}
