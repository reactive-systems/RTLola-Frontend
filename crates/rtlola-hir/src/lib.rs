#![forbid(unused_must_use)] // disallow discarding errors
#![warn(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

pub mod hir;
mod modes;
mod stdlib;
pub mod type_check;

use hir::Hir;
pub use hir::RtLolaHir;
use modes::ast_conversion::TransformationErr;
use modes::dependencies::DependencyErr;
use modes::memory_bounds::MemBoundErr;
use modes::ordering::OrderErr;
use modes::CompletionErr;
pub use modes::{BaseMode, CompleteMode};
use rtlola_parser::RtLolaAst;
use rtlola_reporting::Handler;

pub fn from_ast(ast: RtLolaAst, handler: &Handler) -> Result<Hir<BaseMode>, TransformationErr> {
    Hir::<BaseMode>::from_ast(ast, handler)
}

pub fn fully_analyzed(ast: RtLolaAst, handler: &Handler) -> Result<Hir<CompleteMode>, HirErr> {
    Ok(Hir::<BaseMode>::from_ast(ast, handler)?
        .analyze_dependencies(handler)?
        .check_types(handler)?
        .determine_evaluation_order(handler)?
        .determine_memory_bounds(handler)?
        .finalize(handler)?)
}

#[derive(Debug, Clone)]
pub enum HirErr {
    Ast(TransformationErr),
    Dependency(DependencyErr),
    Type(String),
}

impl From<TransformationErr> for HirErr {
    fn from(e: TransformationErr) -> Self {
        Self::Ast(e)
    }
}

impl From<DependencyErr> for HirErr {
    fn from(e: DependencyErr) -> Self {
        Self::Dependency(e)
    }
}

impl From<OrderErr> for HirErr {
    fn from(_e: OrderErr) -> Self {
        panic!("a non-descript error should never surface to the public interface")
    }
}

impl From<MemBoundErr> for HirErr {
    fn from(_e: MemBoundErr) -> Self {
        panic!("a non-descript error should never surface to the public interface")
    }
}

impl From<CompletionErr> for HirErr {
    fn from(_e: CompletionErr) -> Self {
        panic!("a non-descript error should never surface to the public interface")
    }
}

impl From<String> for HirErr {
    fn from(e: String) -> Self {
        Self::Type(e)
    }
}

impl From<()> for HirErr {
    fn from(_e: ()) -> Self {
        panic!("a non-descript error should never surface to the public interface")
    }
}

#[macro_use]
extern crate rtlola_macros;
