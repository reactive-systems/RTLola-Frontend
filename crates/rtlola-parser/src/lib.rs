//! A parser for RTLola specifications
//!
//! This crate offers functionality to transform a textual representation of an RTLola specification into an abstract syntax tree.  The Ast is not the most convenient data structure for
//! modifying or analyzing a specification; there are other options available, outlined below.
//!
//! # Specification Representations
//! * [RtLolaAst]: The Ast represents the abstract syntax of the specification.  It is obtained by first parsing the specification into a homogenous tree
//!  and then remove concrete syntax fragments irrelevant for the logics of the specification.  Apart from that, the Ast does not provide much functionality.
//!  The only checks performed when creating the Ast concern the correct syntax.  See also: [RtLolaAst], and [parse()].
//! * [RtLolaHir](https://docs.rs/rtlola_hir/struct.RtLolaHir.html): The Hir represents a high-level intermediate representation optimized for analyzability.  It contains more convenient methods than the Ast, enables different
//!  analysis steps and provides their reports.  The Hir traverses several modes representing the level to which it was analyzed and refined.
//!  Its base mode is `RtLolaHir<BaseMode>` and its fully analyzed version is `RtLolaHir<CompleteMode>`.  See also: [rtlola_hir](https://docs.rs/rtlola_hir).
//! * [RtLolaMir](https://docs.rs/rtlola_frontend/struct.RtLolaMir.html): The Mir represents a mid-level intermediate representation optimized for external use such as interpretation and compilation.  It contains several interconnections
//!  enabling easy accesses and additional annotation such as memory bounds for each stream. See also: [rtlola_hir](https://docs.rs/rtlola_hir).
//! As a rule of thumb, if you want to analyze and/or enrich a specification, use the [RtLolaHir](https://docs.rs/rtlola_hir/struct.RtLolaHir.html).  If you only need a convenient representation of the specification for some devious
//! activity such as compiling it into something else, the [RtLolaMir](https://docs.rs/rtlola_frontend/struct.RtLolaMir.html) is the way to go.
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

//! This module provides the functionality needed to parse an RTLola specification into a [RtLolaAst].

mod parse; // Shall not be exposed; use parse function instead.

use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

// Public exports
pub mod ast;
pub use ast::RtLolaAst;
use rtlola_reporting::Handler;

#[derive(Debug, Clone)]
/// The configuration of the parser.
pub struct ParserConfig {
    /// The path to the specification file that should be parsed
    path: Option<PathBuf>,
    /// The specification given as a string
    spec: String,
}

impl ParserConfig {
    /// Reads the specification from the given path and creates a new parser configuration for it.
    pub fn from_path(path_to_spec: PathBuf) -> io::Result<Self> {
        let mut file = File::open(&path_to_spec)?;
        let mut spec = String::new();
        file.read_to_string(&mut spec)?;
        drop(file);
        Ok(ParserConfig {
            path: Some(path_to_spec),
            spec,
        })
    }

    /// Creates a new parser configuration for the given specification.
    pub fn for_string(spec: String) -> Self {
        ParserConfig { path: None, spec }
    }

    /// Invokes the parser on the specification given in the configuration.
    pub fn parse(self) -> Result<RtLolaAst, String> {
        parse(self)
    }

    /// Returns the path of the specification.
    pub fn path(&self) -> &Option<PathBuf> {
        &self.path
    }

    /// Returns the specification of the configuration.
    pub fn spec(&self) -> &str {
        &self.spec
    }
}

/// Invokes the parser with the given configuration.
pub fn parse(cfg: ParserConfig) -> Result<RtLolaAst, String> {
    let handler = if let Some(path) = &cfg.path {
        rtlola_reporting::Handler::new(path.clone(), cfg.spec.clone())
    } else {
        rtlola_reporting::Handler::without_file(cfg.spec.clone())
    };

    let spec = match crate::parse::RTLolaParser::parse(&handler, cfg) {
        Ok(spec) => spec,
        Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        },
    };
    Ok(spec)
}

/// Invokes the parser with the given configuration and uses the provided [Handler] for error reporting.
pub fn parse_with_handler(cfg: ParserConfig, handler: &Handler) -> Result<RtLolaAst, String> {
    let spec = match crate::parse::RTLolaParser::parse(&handler, cfg) {
        Ok(spec) => spec,
        Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        },
    };
    Ok(spec)
}
