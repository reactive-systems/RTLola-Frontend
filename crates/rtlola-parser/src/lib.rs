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
        }
    };
    Ok(spec)
}

/// Invokes the parser with the given configuration and uses the provided [Handler] for error reporting.
pub fn parse_with_handler(cfg: ParserConfig, handler: &Handler) -> Result<RtLolaAst, String> {
    let spec = match crate::parse::RTLolaParser::parse(&handler, cfg) {
        Ok(spec) => spec,
        Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        }
    };
    Ok(spec)
}
