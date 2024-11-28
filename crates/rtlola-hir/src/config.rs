//! Defines the configuration for the analysis stages of the RTLola Hir.

use rtlola_parser::ParserConfig;

/// Represents the configuration for the whole frontend.
///
/// This includes the configuration of the parser, as well as
/// configuration for the analysis stages.
#[derive(Debug)]
pub struct FrontendConfig<'a> {
    parser_config: &'a ParserConfig,
}

impl<'a> From<&'a ParserConfig> for FrontendConfig<'a> {
    fn from(parser_config: &'a ParserConfig) -> Self {
        Self { parser_config }
    }
}

/// Extension to provide additional methods to the ParserConfig.
pub trait ParserConfigExt<'a> {
    /// Returns a reference to the underlying parser config.
    fn parser_config(&self) -> &ParserConfig;
}

impl<'a> ParserConfigExt<'a> for ParserConfig {
    fn parser_config(&self) -> &ParserConfig {
        self
    }
}

impl<'a> FrontendConfig<'a> {
    /// Returns the configuration for the parser
    pub fn parser_config(&self) -> &ParserConfig {
        self.parser_config
    }
}
