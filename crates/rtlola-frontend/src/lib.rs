//! Parser for the RTLola language.

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

mod lowering;
pub mod mir;

use mir::Mir;
use rtlola_hir::{CompleteMode, HirErr};
use rtlola_parser::ParserConfig;

#[cfg(test)]
mod tests;

// Re-export
pub(crate) use rtlola_hir::hir::RtLolaHir as Hir;

pub use crate::mir::RTLolaMIR;

// Replace by more elaborate interface.
/**
Parses a RTLola specification and transforms it to optimize the runtime.

The string passed in as `spec_str` should be the content of the file specified by `filename`.
The filename is only used for printing locations.
See the `FrontendConfig` documentation on more information about the parser options.
*/
pub fn parse(config: ParserConfig) -> Result<RTLolaMIR, FrontEndErr> {
    let hir = parse_to_hir(config)?;
    Ok(Mir::from_hir(hir))
}

/**
Parses a RTLola specification to the high-level intermediate representation.

The string passed in as `spec_str` should be the content of the file specified by `filename`.
The filename is only used for printing locations.
See the `FrontendConfig` documentation on more information about the parser options.
*/
pub(crate) fn parse_to_hir(cfg: ParserConfig) -> Result<Hir<CompleteMode>, FrontEndErr> {
    let handler = if let Some(path) = &cfg.path() {
        rtlola_reporting::Handler::new(path.clone(), String::from(cfg.spec()))
    } else {
        rtlola_reporting::Handler::without_file(String::from(cfg.spec()))
    };
    let spec = rtlola_parser::parse_with_handler(cfg, &handler)?;

    Ok(rtlola_hir::fully_analyzed(spec, &handler)?)
}

#[derive(Debug, Clone)]
pub enum FrontEndErr {
    Parser(String),
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
