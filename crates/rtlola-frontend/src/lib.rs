//! Parser for the RTLola language.

#![forbid(unused_must_use)] // disallow discarding errors
#![warn(
//    missing_docs, //TODO readd when typechecker is stable
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
mod stdlib;

use mir::Mir;
use rtlola_hir::hir::modes::{CompleteMode, IrExprMode};

#[cfg(test)]
mod tests;

// Re-export
pub use crate::mir::RTLolaMIR;
pub(crate) use rtlola_hir::hir::RTLolaHir as Hir;

// Replace by more elaborate interface.
/**
Parses a RTLola specification and transforms it to optimize the runtime.

The string passed in as `spec_str` should be the content of the file specified by `filename`.
The filename is only used for printing locations.
See the `FrontendConfig` documentation on more information about the parser options.
*/
pub fn parse(filename: &str, spec_str: &str, config: rtlola_parser::FrontendConfig) -> Result<RTLolaMIR, String> {
    let hir = parse_to_hir(filename, spec_str, config);
    match hir {
        Err(_) => Err("Analysis failed due to errors in the specification".to_string()),
        Ok(hir) => Ok(Mir::from_hir(hir)),
    }
}

/**
Parses a RTLola specification to the high-level intermediate representation.

The string passed in as `spec_str` should be the content of the file specified by `filename`.
The filename is only used for printing locations.
See the `FrontendConfig` documentation on more information about the parser options.
*/
pub(crate) fn parse_to_hir(
    filename: &str,
    spec_str: &str,
    config: rtlola_parser::FrontendConfig,
) -> Result<Hir<CompleteMode>, String> {
    let spec = rtlola_parser::parse(filename, spec_str, config)?;

    let handler = rtlola_reporting::Handler::new(std::path::PathBuf::from(filename), spec_str.into());

    Ok(Hir::<IrExprMode>::transform_expressions(spec, &handler, &config)
        .map_err(|e| format!("error in expression transformation: {:?}", e))?
        .build_dependency_graph()
        .map_err(|e| format!("error in dependency analysis: {:?}", e))?
        .type_check(&handler)?
        .build_evaluation_order()
        .compute_memory_bounds()
        .finalize())
    // let analysis_result = analysis::analyze(&spec, &handler, config);
    // analysis_result
    //     .map(|report| hir::RTLolaHIR::<FullInformationHirMode>::new(&spec, &report))
    //     .map_err(|_| "Analysis failed due to errors in the specification".to_string())
}
