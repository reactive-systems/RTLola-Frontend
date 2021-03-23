#![allow(dead_code)]

pub mod hir;
mod modes;
pub mod type_check;

pub use hir::RtLolaHir;
pub use modes::{
    CompleteMode, DepAnaMode, DepAnaTrait, IrExprMode, IrExprTrait, MemBoundMode, MemBoundTrait, OrderedMode,
    OrderedTrait,
};

#[macro_use]
extern crate rtlola_macros;
