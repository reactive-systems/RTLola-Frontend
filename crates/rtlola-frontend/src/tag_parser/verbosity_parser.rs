//! Contains tag parser and validators to support verbosity and debug tags in the specification
//! to use by backends for logging purposes.

use std::collections::{HashMap, HashSet};

use rtlola_hir::hir::StreamReference;
use rtlola_reporting::{Diagnostic, RtLolaError};

use super::{TagParser, TagValidator};
use crate::RtLolaMir;

#[derive(Debug, Copy, Clone)]
/// A tag validator parsing annotated stream verbosities
pub struct VerbosityParser;

/// Represents the annotated verbosity of a stream
#[derive(Debug, Clone, Copy)]
pub enum StreamVerbosity {
    /// The stream is tagged as verbosity level `streams`
    Streams,
    /// The stream is tagged as verbosity level `outputs`
    Outputs,
    /// The stream is tagged as verbosity level `public`
    Public,
    /// The stream is tagged as verbosity level `warning`
    Warnings,
    /// The stream is tagged as verbosity level `violation`
    Violations,
}

impl TagParser for VerbosityParser {
    type GlobalTags = ();
    type LocalTags = Option<StreamVerbosity>;

    fn parse_global(
        &self,
        _global_tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::GlobalTags, RtLolaError> {
        Ok(())
    }

    fn parse_local(
        &self,
        sr: StreamReference,
        tags: &HashMap<String, Option<String>>,
        mir: &RtLolaMir,
    ) -> Result<Self::LocalTags, RtLolaError> {
        let tags = tags
            .iter()
            .filter_map(|(key, value)| match (key.as_str(), value) {
                ("verbosity", Some(name)) => Some(match name.as_str() {
                    "streams" => Ok(StreamVerbosity::Streams),
                    "outputs" => Ok(StreamVerbosity::Outputs),
                    "public" => Ok(StreamVerbosity::Public),
                    "warnings" => Ok(StreamVerbosity::Warnings),
                    "violations" => Ok(StreamVerbosity::Violations),
                    other => Err(Diagnostic::error(&format!(
                        "Annotated unexpected verbosity {other} on stream {}",
                        mir.stream(sr).name()
                    ))),
                }),
                ("verbosity", None) => Some(Err(Diagnostic::error(&format!(
                    "Missing verbosity value on annotation on stream {}",
                    mir.stream(sr).name()
                )))),
                ("warning", None) => Some(Ok(StreamVerbosity::Warnings)),
                ("warning", Some(_)) => panic!(),
                ("violation", None) => Some(Ok(StreamVerbosity::Violations)),
                ("violation", Some(_)) => panic!(),
                ("public", None) => Some(Ok(StreamVerbosity::Public)),
                ("public", Some(_)) => panic!(),
                (_, _) => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        match tags.len() {
            0 => Ok(None),
            1 => Ok(Some(tags[0])),
            2.. => Err(Diagnostic::error(&format!(
                "Specified multiple verbosities on stream {}",
                mir.stream(sr).name()
            ))
            .into()),
        }
    }
}

impl TagValidator for VerbosityParser {
    fn supported_tags<'a>(&self, _mir: &'a RtLolaMir) -> (HashSet<&'a str>, HashSet<&'a str>) {
        (
            HashSet::new(),
            vec!["verbosity", "warning", "violation", "public"]
                .into_iter()
                .collect(),
        )
    }
}

#[derive(Debug, Copy, Clone)]
/// A tag validator parsing a debug annotation
pub struct DebugParser;

impl TagParser for DebugParser {
    type GlobalTags = ();
    type LocalTags = bool;

    fn parse_global(
        &self,
        _global_tags: &HashMap<String, Option<String>>,
        _mir: &RtLolaMir,
    ) -> Result<Self::GlobalTags, RtLolaError> {
        Ok(())
    }

    fn parse_local(
        &self,
        sr: StreamReference,
        tags: &HashMap<String, Option<String>>,
        mir: &RtLolaMir,
    ) -> Result<Self::LocalTags, RtLolaError> {
        match tags.get("debug") {
            Some(None) => Ok(true),
            None => Ok(false),
            Some(Some(_)) => Err(Diagnostic::error(&format!(
                "The debug tag on stream {} received an unexpected value",
                mir.stream(sr).name()
            ))
            .into()),
        }
    }
}

impl TagValidator for DebugParser {
    fn supported_tags<'a>(&self, _mir: &'a RtLolaMir) -> (HashSet<&'a str>, HashSet<&'a str>) {
        (HashSet::new(), vec!["debug"].into_iter().collect())
    }
}
