//! Parser for the Lola language.

#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;
extern crate clap;

pub mod app;
mod ast;
mod parse;
mod print;

// Re-export on the root level
pub use ast::{LanguageSpec, LolaSpec};
