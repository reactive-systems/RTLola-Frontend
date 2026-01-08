//! Defines the configuration for the analysis stages of the RTLola Hir.

use rtlola_parser::ParserConfig;

pub use crate::modes::privacy::PrivacyHeuristic;

/// Represents the configuration for the whole frontend.
///
/// This includes the configuration of the parser, as well as
/// configuration for the analysis stages.
#[derive(Debug)]
pub struct FrontendConfig<'a> {
    parser_config: &'a ParserConfig,
    memory_bound_mode: MemoryBoundMode,
    privacy_parameter: Option<f64>,
    privacy_heuristic: PrivacyHeuristic,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
/// The way the memory bound is computed.
pub enum MemoryBoundMode {
    #[default]
    /// All values, including values that only exist during a cycle, are counted towards the memory.
    /// (all streams have a memory bound of at least 1).
    Static,
    /// Counts only values that need to be retained between cycles
    /// (streams with direct sync accesses have memory bound of 0).
    Dynamic,
}

impl<'a> From<&'a ParserConfig> for FrontendConfig<'a> {
    fn from(parser_config: &'a ParserConfig) -> Self {
        Self {
            parser_config,
            memory_bound_mode: MemoryBoundMode::default(),
            privacy_parameter: None,
            privacy_heuristic: PrivacyHeuristic::Inputs,
        }
    }
}

/// Extension to provide additional methods to the ParserConfig.
pub trait ParserConfigExt<'a> {
    /// Specifies whether to compute the memory bound in dynamic or static way.
    fn with_memory_bound_mode(&'a self, memory_bound_mode: MemoryBoundMode) -> FrontendConfig<'a>;

    /// Returns a reference to the underlying parser config.
    fn parser_config(&self) -> &ParserConfig;

    /// Make the specification private with the given privacy parameter
    fn with_privacy_parameter(&'a self, parameter: f64) -> FrontendConfig<'a>;
}

impl<'a> ParserConfigExt<'a> for ParserConfig {
    fn with_memory_bound_mode(&'a self, memory_bound_mode: MemoryBoundMode) -> FrontendConfig<'a> {
        FrontendConfig::from(self).with_memory_bound_mode(memory_bound_mode)
    }

    fn parser_config(&self) -> &ParserConfig {
        self
    }

    fn with_privacy_parameter(&'a self, parameter: f64) -> FrontendConfig<'a> {
        FrontendConfig::from(self).with_privacy_parameter(parameter)
    }
}

impl<'a> FrontendConfig<'a> {
    fn with_memory_bound_mode(self, memory_bound_mode: MemoryBoundMode) -> FrontendConfig<'a> {
        Self {
            memory_bound_mode,
            ..self
        }
    }

    pub fn with_privacy_parameter(self, parameter: f64) -> FrontendConfig<'a> {
        Self {
            privacy_parameter: Some(parameter),
            ..self
        }
    }

    pub fn with_privacy_heuristic(self, heuristic: PrivacyHeuristic) -> FrontendConfig<'a> {
        Self {
            privacy_heuristic: heuristic,
            ..self
        }
    }

    /// Returns the configuration for the parser
    pub fn parser_config(&self) -> &ParserConfig {
        self.parser_config
    }

    pub fn privacy_parameter(&self) -> Option<f64> {
        self.privacy_parameter
    }

    pub fn privacy_heuristic(&self) -> PrivacyHeuristic {
        self.privacy_heuristic
    }
}

impl FrontendConfig<'_> {
    pub(crate) fn memory_bound_mode(&self) -> MemoryBoundMode {
        self.memory_bound_mode
    }
}
