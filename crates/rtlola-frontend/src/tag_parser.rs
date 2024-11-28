//! This module provides a parser to interpret the annotated tags of a specification
//! as a custom type.
//!
//! Each parser should implement the [TagParser] trait.
//! The main entry points for applying the parser is the [RtLolaMir::parse_tags] method,
//! which receives a list of parsers and returns a list of parse results.
//! The [RtLolaMir::check_tags] method allows for validating that all annotated tags are
//! validated by any parser. This allows for typos to be detected.

pub mod all;
pub mod verbosity_parser;

use std::collections::{HashMap, HashSet};

use rtlola_hir::hir::StreamReference;
use rtlola_reporting::{Diagnostic, RtLolaError};

use super::RtLolaMir;

/// Represents a parser supporting a subset of all tags annotated to the specification
pub trait TagParser: TagValidator {
    /// The type representing the result of parsing global tags
    type GlobalTags;
    /// The type representing the result of parsing local tags (on streams)
    type LocalTags;

    /// Parses the global tags to `Self::GlobalTags`
    fn parse_global(
        &self,
        global_tags: &HashMap<String, Option<String>>,
        mir: &RtLolaMir,
    ) -> Result<Self::GlobalTags, RtLolaError>;

    /// Parses the tags annotated to a stream to `Self::LocalTags`
    fn parse_local(
        &self,
        sr: StreamReference,
        tags: &HashMap<String, Option<String>>,
        mir: &RtLolaMir,
    ) -> Result<Self::LocalTags, RtLolaError>;
}

/// Specifies for a [TagParser] the tag-keys that are validated by the Parser.
/// All other keys are ignored.
pub trait TagValidator {
    /// Returns the global/local tags that are supported by the parser.
    fn supported_tags<'a>(&self, mir: &'a RtLolaMir) -> (HashSet<&'a str>, HashSet<&'a str>);
}

/// The result after applying a [TagParser] to a specification
#[derive(Debug)]
pub struct ParseResult<GlobalTags, LocalTags> {
    /// The representation of the global tags
    global_tags: GlobalTags,
    /// A mapping from streams to the corresponding representation of the local tags
    local_tags: HashMap<StreamReference, LocalTags>,
}

impl<GlobalTags, LocalTags> ParseResult<GlobalTags, LocalTags> {
    /// Returns the representation returned by the parser for the global tags
    pub fn global_tags(&self) -> &GlobalTags {
        &self.global_tags
    }

    /// Returns the representation returned by the parser for the given stream (if it exists)
    pub fn local_tags(&self, sr: StreamReference) -> Option<&LocalTags> {
        self.local_tags.get(&sr)
    }
}

impl RtLolaMir {
    /// Checks and applies the given parser to the specification to return a list of [ParseResult]'s.
    pub fn parse_tags<T: TagParser>(&self, p: T) -> Result<ParseResult<T::GlobalTags, T::LocalTags>, RtLolaError> {
        let global_tags = p.parse_global(&self.global_tags, self)?;
        let local_tags_iter = self
            .all_streams()
            .map(|sr| Ok((sr, p.parse_local(sr, self.stream(sr).tags(), self)?)));
        let local_tags = RtLolaError::collect(local_tags_iter)?;

        Ok(ParseResult {
            global_tags,
            local_tags,
        })
    }

    /// Checks whether the specification contains tags that are not handled by any parser
    pub fn validate_tags(&self, parser: &[&dyn TagValidator]) -> Result<(), RtLolaError> {
        let global_keys = self.global_tags.keys().map(|k| k.as_str()).collect::<HashSet<_>>();
        let local_keys = self
            .all_streams()
            .flat_map(|sr| self.stream(sr).tags().keys().map(|k| k.as_str()))
            .collect::<HashSet<_>>();
        let (unused_gt, unused_lt) =
            parser
                .iter()
                .fold((global_keys.clone(), local_keys.clone()), |(mut gt, mut lt), p| {
                    let (pgt, plt) = p.supported_tags(self);
                    gt = &gt - &pgt;
                    lt = &lt - &plt;
                    (gt, lt)
                });
        if !unused_gt.is_empty() {
            return Err(Diagnostic::error(&format!("Unused global tags: {unused_gt:?}")).into());
        }
        if !unused_lt.is_empty() {
            return Err(Diagnostic::error(&format!("Unused local tags: {unused_lt:?}")).into());
        }
        Ok(())
    }

    /// Returns the keys of all global tags in the specification
    pub fn all_global_tags(&self) -> HashSet<&str> {
        self.global_tags.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the keys of all local tags used anywhere in the specification
    pub fn all_local_tags(&self) -> HashSet<&str> {
        self.all_streams()
            .flat_map(|sr| self.stream(sr).tags().keys())
            .map(|s| s.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;

    use super::all::{AllowAll, DenyAll};
    use crate::parse;
    use crate::tag_parser::verbosity_parser::{DebugParser, VerbosityParser};

    #[test]
    fn check_tags1() {
        let spec = "
		#![supported,unsupported]	
		input a : UInt64\n\
		#[tag,unsupported=\"value\", unsupported2]
		output b := a + 1\n\
		trigger b > 10";
        let mir = parse(&ParserConfig::for_string(spec.into())).unwrap();
        assert!(mir.validate_tags(&[&AllowAll]).is_ok());
        assert!(mir.validate_tags(&[&DenyAll]).is_err());
    }

    #[test]
    fn check_tags_verbosity() {
        let spec = "
		input a : UInt64\n\
		#[verbosity=\"public\"]
		output b := a + 1\n\
		#[warning]
		trigger b > 10";
        let mir = parse(&ParserConfig::for_string(spec.into())).unwrap();
        mir.validate_tags(&[&VerbosityParser]).unwrap();
    }

    #[test]
    fn check_tags_verbosity2() {
        let spec = "
		input a : UInt64\n\
		#[verbosity=\"public\"]
		output b := a + 1\n\
		#[warnig] // <-- typo here
		trigger b > 10";
        let mir = parse(&ParserConfig::for_string(spec.into())).unwrap();
        assert!(mir.validate_tags(&[&VerbosityParser]).is_err());
    }

    #[test]
    fn check_tags_verbosity_and_debug() {
        let spec = "
		input a : UInt64\n\
		#[verbosity=\"public\", debug]
		output b := a + 1\n\
		#[debug]
		trigger b > 10";
        let mir = parse(&ParserConfig::for_string(spec.into())).unwrap();
        mir.validate_tags(&[&VerbosityParser, &DebugParser]).unwrap();
    }

    #[test]
    fn check_tags_verbosity_and_debug2() {
        let spec = "
		input a : UInt64\n\
		#[verbosity=\"public\", debug]
		output b := a + 1\n\
		#[debg] // <-- typo here
		trigger b > 10";
        let mir = parse(&ParserConfig::for_string(spec.into())).unwrap();
        assert!(mir.validate_tags(&[&VerbosityParser, &DebugParser]).is_err());
    }
}
