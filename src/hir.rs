/*!
This module describes the high level intermediate representation of a specification. This representation is used to transform the specification, e.g. to optimize or to introduce syntactic sugar.

The module occurs in different modes, adding different information to the intermediate representation.
*/
use crate::common_ir::StreamReference as SRef;
use crate::common_ir::*;
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

/// Represents an input stream in an RTLola specification.
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct Input {
    /// The name of the stream.
    name: String,
    /// The reference pointing to this stream.
    sr: SRef,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Output {
    /// The name of the stream.
    name: String,
    /// The reference pointing to this stream.
    sr: StreamReference,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Trigger {
    name: String,
    message: String,
    sr: SRef,
}

impl Trigger {
    fn new(_name: Option<parse::Ident>, _msg: Option<String>, _sr: SRef) -> Self {
        let _name = unimplemented!();
        // Self { name, message: msg.unwrap_or_else(String::new), sr }
    }
}
