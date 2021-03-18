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

pub mod ast;
pub mod common_ir;
pub mod hir;
pub mod mir;
pub mod naming;
pub mod parse;
pub mod reporting;
mod stdlib;
// mod transformations;
mod ty;
mod tyc;

#[macro_use]
extern crate rtlola_macros;
#[cfg(test)]
mod tests;

use hir::{
    modes::{CompleteMode, IrExprMode},
    Hir,
};

// Re-export
pub use crate::mir::RTLolaMIR;
pub use crate::ty::TypeConfig;

/**
Hold the configuration of the frontend
*/
#[derive(Debug, Clone, Copy)]
pub struct FrontendConfig {
    /**
    Several options regarding the type-system. See the `TypeConfig` documentation for more information.
    */
    pub ty: TypeConfig,
    /**
    A flag whether streams can parameterized.
    */
    pub allow_parameters: bool,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self { ty: TypeConfig::default(), allow_parameters: true }
    }
}

// Replace by more elaborate interface.
/**
Parses a RTLola specification and transforms it to optimize the runtime.

The string passed in as `spec_str` should be the content of the file specified by `filename`.
The filename is only used for printing locations.
See the `FrontendConfig` documentation on more information about the parser options.
*/
pub fn parse(filename: &str, spec_str: &str, config: FrontendConfig) -> Result<RTLolaMIR, String> {
    let hir = parse_to_hir(filename, spec_str, config);
    match hir {
        Err(_) => Err("Analysis failed due to errors in the specification".to_string()),
        Ok(hir) => Ok(hir.lower()),
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
    config: FrontendConfig,
) -> Result<Hir<CompleteMode>, String> {
    let handler = reporting::Handler::new(std::path::PathBuf::from(filename), spec_str.into());

    let spec = match crate::parse::parse(&spec_str, &handler, config) {
        Ok(spec) => spec,
        Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        }
    };
    Ok(Hir::<IrExprMode>::transform_expressions(spec, &handler, &config)
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
