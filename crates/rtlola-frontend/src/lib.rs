//! The front-end of the RTLola Monitoring Framework
//!
//! This crate offers functionality to transform a textual representation of an RTLola specification into one of three tree-shaped representation and perform a variety of checks.
//!
//! # Specification Representations
//! * [RtLolaAst]: The Ast represents the abstract syntax of the specification.  It is obtained by first parsing the specification into a homogenous tree
//!  and then remove concrete syntax fragments irrelevant for the logics of the specification.  Apart from that, the Ast does not provide much functionality.
//!  The only checks performed when creating the Ast concern the correct syntax.  See also: [rtlola_parser], [RtLolaAst], and [parse_to_ast].
//! * [RtLolaHir]: The Hir represents a high-level intermediate representation optimized for analyzability.  It contains more convenient methods than the Ast, enables different
//!  analysis steps and provides their reports.  The Hir traverses several [rtlola_hir::hir::HirMode]s representing the level to which it was analyzed and refined.
//!  Its base mode is `RtLolaHir<BaseMode>` and its fully analyzed version is `RtLolaHir<CompleteMode>`.  See also: [rtlola_hir], [RtLolaHir], [parse_to_base_hir], and [parse_to_base_hir].
//! * [RtLolaMir]: The Mir prepresents a mid-level intermediate reprensetation optimized for extenal use such as interpretation and compilation.  It contains several interconnections
//!  enabling easy accesses and additional annotation such as memory bounds for each stream. See also: [RtLolair], [parse].
//! As a rule of thumb, if you want to analyze and/or enrich a specification, use the [RtLolaHir].  If you only need a convenient representation of the specification for some devious
//! activity such as compiling it into something else, the [RtLolaMir] is the way to go.
//!
//! # Modules
//! * [mir] Contains anything related to the [RtLolaMir].

#![forbid(unused_must_use)]
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

mod lowering;
pub mod mir;

use mir::Mir;
use rtlola_hir::{BaseMode, CompleteMode, HirErr};
use rtlola_parser::{ParserConfig, RtLolaAst};

#[cfg(test)]
mod tests;

pub(crate) use rtlola_hir::hir::{RtLolaHir as Hir, RtLolaHir};
use rtlola_reporting::Handler;

pub use crate::mir::RtLolaMir;

/// Attempts to parse a textual specification into an [RtLolaMir].
///
/// The specification is wrapped into a [ParserConfig] and can either be a string or a path to a specification file.
///
/// # Fail
/// Fails if either the parsing was unsuccessful due to parsing errors such as incorrect syntax (cf. [FrontEndErr::Parser]) or an analysis failed
/// due to a semantic error such as inconsistent types or unknown identifiers (cf. [FrontEndErr::Analysis] and [HirErr]).
pub fn parse(config: ParserConfig) -> Result<RtLolaMir, FrontEndErr> {
    let hir = parse_to_hir(config)?;
    Ok(Mir::from_hir(hir))
}

/// Attempts to parse a textual specification into a fully analyzed [RtLolaHir]<[CompleteMode]>.
///
/// The specification is wrapped into a [ParserConfig] and can either be a string or a path to a specification file.
///
/// # Fail
/// Fails if either the parsing was unsuccessful due to parsing errors such as incorrect syntax (cf. [FrontEndErr::Parser]) or an analysis failed
/// due to a semantic error such as inconsistent types or unknown identifiers (cf. [FrontEndErr::Analysis] and [HirErr]).
pub fn parse_to_final_hir(cfg: ParserConfig) -> Result<Hir<CompleteMode>, FrontEndErr> {
    let handler = create_handler(&cfg);
    let spec = rtlola_parser::parse_with_handler(cfg, &handler)?;

    Ok(rtlola_hir::fully_analyzed(spec, &handler)?)
}

/// Attempts to parse a textual specification into a basic [RtLolaHir]<[BaseMode]>.
///
/// The specification is wrapped into a [ParserConfig] and can either be a string or a path to a specification file.
///
/// # Fail
/// Fails if either the parsing was unsuccessful due to parsing errors such as incorrect syntax (cf. [FrontEndErr::Parser]) or the initial analysis failed
/// due occurences of unknown identifiers (cf. [FrontEndErr::Analysis] and [HirErr::Ast], specifically [rtlola_hir::TransformationErr]).
pub fn parse_to_base_hir(cfg: ParserConfig) -> Result<Hir<BaseMode>, FrontEndErr> {
    let handler = create_handler(&cfg);
    let spec = rtlola_parser::parse_with_handler(cfg, &handler)?;

    Ok(rtlola_hir::fully_analyzed(spec, &handler)?)
}

/// Attempts to parse a textual specification into an [RtLolaAst].
///
/// The specification is wrapped into a [ParserConfig] and can either be a string or a path to a specification file.
///
/// # Fail
/// Fails if the parsing was unsuccessful due to parsing errors such as incorrect syntax (cf. [FrontEndErr::Parser]).
pub fn parse_to_ast(cfg: ParserConfig) -> Result<RtLolaAst, FrontEndErr> {
    let handler = create_handler(&cfg);
    let spec = rtlola_parser::parse_with_handler(cfg, &handler)?;
}

fn create_handler(cfg: &ParserConfig) -> Handler {
    if let Some(path) = &cfg.path() {
        rtlola_reporting::Handler::new(path.clone(), String::from(cfg.spec()))
    } else {
        rtlola_reporting::Handler::without_file(String::from(cfg.spec()))
    }
}

/// A wrapper for the different kinds of errors that can occur in the front-end.
#[derive(Debug, Clone)]
pub enum FrontEndErr {
    /// Indicates a parsing error and provides a more detailed error description.  The error originates in [rtlola_parser].
    Parser(String),
    /// Indicates an analysis error and provides a more detailed error description (cf. [HirErr]). The error originates in [rtlola_hir].
    Analysis(HirErr),
}

impl From<String> for FrontEndErr {
    fn from(s: String) -> FrontEndErr {
        Self::Parser(s)
    }
}

impl From<HirErr> for FrontEndErr {
    fn from(e: HirErr) -> FrontEndErr {
        Self::Analysis(e)
    }
}
