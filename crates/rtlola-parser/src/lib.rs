mod parse; // Shall not be exposed; use parse function instead.

use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

// Public exports
pub mod ast;
pub use ast::RTLolaAst;

#[derive(Debug, Clone)]
pub struct ParserConfig {
    // Path to spec file.
    path: Option<PathBuf>,
    spec: String,
}

impl ParserConfig {
    pub fn from_path(path_to_spec: PathBuf) -> io::Result<Self> {
        let mut file = File::open(&path_to_spec)?;
        let mut spec = String::new();
        file.read_to_string(&mut spec)?;
        drop(file);
        Ok(ParserConfig { path: Some(path_to_spec), spec })
    }
    pub fn for_string(spec: String) -> Self {
        ParserConfig { path: None, spec }
    }
    pub fn parse(self) -> Result<RTLolaAst, String> {
        parse(self)
    }
}

pub fn parse(cfg: ParserConfig) -> Result<RTLolaAst, String> {
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
