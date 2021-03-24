pub mod hir;
mod modes;
mod stdlib;
pub mod type_check;

use hir::Hir;
use rtlola_parser::RTLolaAst;

pub use hir::RtLolaHir;
pub use modes::dependencies::DependencyErr;
pub use modes::ir_expr::TransformationErr;
pub use modes::{
    CompleteMode, DepAnaMode, DepAnaTrait, HirStage, IrExprMode, IrExprTrait, MemBoundMode, MemBoundTrait, OrderedMode,
    OrderedTrait,
};
use rtlola_reporting::Handler;

pub fn from_ast(ast: RTLolaAst, handler: &Handler) -> Result<Hir<IrExprMode>, TransformationErr> {
    Hir::<IrExprMode>::from_ast(ast, handler)
}

pub fn fully_analyzed(ast: RTLolaAst, handler: &Handler) -> Result<Hir<CompleteMode>, HirErr> {
    Ok(Hir::<IrExprMode>::from_ast(ast, handler)?
        .analyze_dependencies(handler)?
        .check_types(handler)?
        .determine_evaluation_order(handler)?
        .determine_memory_bounds(handler)?
        .finalize(handler)?)
}

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
