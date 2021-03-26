//! The high-level intermediate representation of the RTLola Monitoring Framework
//!
//! This crate offers functionality to transform a textual representation of an RTLola specification into an abstract syntax tree.  The Ast is not the most convenient data structure for
//! modifying or analyzing a specification; there are other options available, outlined below.
//!
//! # Specification Representations
//! * [RtLolaAst]: The Ast represents the abstract syntax of the specification.  It is obtained by first parsing the specification into a homogenous tree
//!  and then remove concrete syntax fragments irrelevant for the logics of the specification.  Apart from that, the Ast does not provide much functionality.
//!  The only checks performed when creating the Ast concern the correct syntax.
//! * [https://docs.rs/rtlola_hir/struct.RTLolaHir.html](RtLolaHir): The Hir represents a high-level intermediate representation optimized for analyzability.  It contains more convenient methods than the Ast, enables different
//!  analysis steps and provides their reports.  The Hir traverses several modes representing the level to which it was analyzed and refined.
//!  Its base mode is `RtLolaHir<BaseMode>` and its fully analyzed version is `RtLolaHir<CompleteMode>`.  See also: [https://docs.rs/rtlola_hir/](RTLola Hir Crate).
//! * [https://docs.rs/rtlola_frontend/struct.RTLolaMir.html](RtLolaMir): The Mir represents a mid-level intermediate representation optimized for external use such as interpretation and compilation.  It contains
//!  several interconnections enabling easy accesses and additional annotation such as memory bounds for each stream. See also: [https://docs.rs/rtlola_frontend/](RTLola FrontEnd Crate).
//! As a rule of thumb, if you want to analyze and/or enrich a specification, use the [RtLolaHir].  If you only need a convenient representation of the specification for some devious
//! activity such as compiling it into something else, the [RtLolaMir] is the way to go.
//!
//! # Modules
//! * [ast] Contains anything related to the [RtLolaAst].

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
mod type_check;

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

/// Transforms a [RtLolaAst] into the [RtLolaHir](crate::hir::RtLolaHir).
///
/// For a given specification the [parse](rtlola_parser::parse) function parses the input into the [RtLolaAst].
/// The [RtLolaHir](crate::hir::RtLolaHir) is the result of the first transformation and is used for all following analyses and transformations.
pub fn from_ast(ast: RtLolaAst, handler: &Handler) -> Result<Hir<BaseMode>, TransformationErr> {
    Hir::<BaseMode>::from_ast(ast, handler)
}

/// Transforms a [RtLolaAst] into the [RtLolaHir](crate::hir::RtLolaHir) and completes all mode transformations.
///
/// The [RtLolaAst] can be obtained by [parse](rtlola_parser::parse)  and its sibling functions.
/// Analyses a performed sequential in the following order:
/// - [ast conversion](crate::hir::RtLolaHir::<BaseMode>::from_ast)
/// - Dependency analysis ([see](crate::hir::RtLolaHir::<TypeMode>::determine_evaluation_order)).
/// - Type analysis ([see](crate::hir::RtLolaHir::<DepAnaMode>::check_types)):
/// - Layer analysis ([see](crate::hir::RtLolaHir::<TypedMode>::determine_evaluation_order)):
/// - Memory analysis ([see](crate::hir::RtLolaHir::<OrderedMode>::determine_memory_bounds)):
///
/// This function returns the fully analysed [RtLolaHir](crate::hir::RtLolaHir)  which can be lowered into the [Mir](rtlola-frontend::Mir).
pub fn fully_analyzed(ast: RtLolaAst, handler: &Handler) -> Result<Hir<CompleteMode>, HirErr> {
    Ok(Hir::<BaseMode>::from_ast(ast, handler)?
        .analyze_dependencies(handler)?
        .check_types(handler)?
        .determine_evaluation_order(handler)?
        .determine_memory_bounds(handler)?
        .finalize(handler)?)
}

/// This [HirErr] is returned by [from_ast] or [fully_analyzed] in the case of an error during the a transformation.
///
/// Each variant contains the more detailed error of the transformation stage.
/// See [TransformationErr], [DependencyErr] and the error emitting by the [Handler].
#[derive(Debug, Clone)]
pub enum HirErr {
    /// Contains an [TransformationErr] occurred during [from_ast].
    Ast(TransformationErr),
    /// Contains an [DependencyErr] occurred during [Dependency analysis](crate::hir::RtLolaHir::<TypeMode>::determine_evaluation_order).
    Dependency(DependencyErr),
    /// Contains an information about an type error, occurred during [Type analysis](crate::hir::RtLolaHir::<DepAnaMode>::check_types).
    /// For detailed information look at the [Handler] error report.
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
