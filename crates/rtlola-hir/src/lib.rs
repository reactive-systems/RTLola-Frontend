//! The high-level intermediate representation of the RTLola Monitoring Framework
//!
//! This crate offers functionality to transform the abstract syntax tree (See [RtLolaAst]) of an RTLola specification into a high-level intermediate representation.
//! It contains more convenient methods than the Ast, enables different analysis steps and provides their reports.  The Hir traverses several modes representing the level to which it was analyzed and refined.
//!
//! # HIR Modes
//! * `RtLolaHir<BaseMode>` is the base mode of the Hir. In this state, the hir contains multiple convenience methods to work with the specification.
//! * `RtLolaHir<DepAnaMode>` additionally features the dependency analysis.
//! * `RtLolaHir<TypedMode>` annotates the streams with value and pacing type information.
//! * `RtLolaHir<OrderedMode>` orders the streams into layers of streams which can be evaluated at the same time.
//! * `RtLolaHir<MemBoundMode>` enriches the streams with their memory requirements.
//! * `RtLolaHir<CompleteMode>` finalizes the Hir to its fully analyzed state.
//! Refer to [RtLolaHir] for more details.

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

use serde::{Deserialize, Serialize};

pub mod hir;
mod modes;
mod stdlib;
mod type_check;

use hir::Hir;
pub use hir::RtLolaHir;
pub use modes::{BaseMode, CompleteMode};
use rtlola_parser::RtLolaAst;
use rtlola_reporting::RtLolaError;

/// Transforms a [RtLolaAst] into the [RtLolaHir](crate::hir::RtLolaHir).
///
/// For a given specification the [parse](rtlola_parser::parse) function parses the input into the [RtLolaAst].
/// The [RtLolaHir](crate::hir::RtLolaHir) is the result of the first transformation and is used for all following analyses and transformations.
pub fn from_ast(ast: RtLolaAst) -> Result<Hir<BaseMode>, RtLolaError> {
    Hir::<BaseMode>::from_ast(ast)
}

/// Transforms a [RtLolaAst] into the [RtLolaHir](crate::hir::RtLolaHir) and completes all mode transformations.
///
/// The [RtLolaAst] can be obtained by [parse](rtlola_parser::parse)  and its sibling functions.
/// Analyses are performed sequentially in the following order:
/// - Initial conversion (see [from_ast])
/// - Dependency analysis (see [determine_evaluation_order](crate::hir::RtLolaHir::<TypeMode>::determine_evaluation_order)).
/// - Type analysis (see [check_types](crate::hir::RtLolaHir::<DepAnaMode>::check_types)):
/// - Layer analysis (see [determine_evaluation_order](crate::hir::RtLolaHir::<TypedMode>::determine_evaluation_order)):
/// - Memory analysis (see [determine_memory_bounds](crate::hir::RtLolaHir::<OrderedMode>::determine_memory_bounds)):
///
/// This function returns the fully analysed [RtLolaHir](crate::hir::RtLolaHir)  which can be lowered into the [Mir](rtlola-frontend::Mir).
pub fn fully_analyzed(ast: RtLolaAst) -> Result<Hir<CompleteMode>, RtLolaError> {
    Hir::<BaseMode>::from_ast(ast)?
        .analyze_dependencies()?
        .check_types()?
        .determine_evaluation_order()?
        .determine_memory_bounds()?
        .finalize()
}

#[macro_use]
extern crate rtlola_macros;
