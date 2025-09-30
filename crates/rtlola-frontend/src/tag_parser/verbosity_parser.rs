//! Contains tag parser and validators to support verbosity and debug tags in the specification
//! to use by backends for logging purposes.

use std::collections::{HashMap, HashSet};

use rtlola_hir::hir::OutputKind;
use rtlola_reporting::{Diagnostic, RtLolaError};

use super::{TagParser, TagValidator};
use crate::{mir::StreamReference, RtLolaMir};

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
    type LocalTags = StreamVerbosity;

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
        #[cfg(feature = "spanned")]
        let mut verbosity_tag_spans = Vec::new();
        let tags = tags
            .iter()
            .filter_map(|(key, value)| {
                let verbosity = match (key.as_str(), value) {
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
                    #[cfg(feature = "spanned")]
                    ("verbosity", None) => Some(Err(Diagnostic::error(&format!(
                        "Missing verbosity value on annotation on stream {}",
                        mir.stream(sr).name()
                    ))
                    .add_span_with_label(
                        mir.stream(sr).tags_span()["verbosity"],
                        Some("Found tag here"),
                        true,
                    ))),
                    #[cfg(not(feature = "spanned"))]
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
                };
                #[cfg(feature = "spanned")]
                if verbosity.is_some() {
                    verbosity_tag_spans.push(mir.stream(sr).tags_span()[key]);
                }
                verbosity
            })
            .collect::<Result<Vec<_>, _>>()?;
        match tags.len() {
            0 => match sr {
                StreamReference::In(_) => Ok(StreamVerbosity::Streams),
                StreamReference::Out(_) => match mir.output(sr).kind {
                    OutputKind::NamedOutput(_) => Ok(StreamVerbosity::Outputs),
                    OutputKind::Trigger(_) => Ok(StreamVerbosity::Violations),
                },
            },
            1 => Ok(tags[0]),
            #[cfg(feature = "spanned")]
            2.. => {
                let mut e = Diagnostic::error(&format!(
                    "Specified multiple verbosities on stream {}",
                    mir.stream(sr).name()
                ));
                for span in verbosity_tag_spans {
                    e = e.add_span_with_label(span, Some("Found verbosity annotation here"), true);
                }
                Err(e.into())
            }
            #[cfg(not(feature = "spanned"))]
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
            #[cfg(feature = "spanned")]
            Some(Some(_)) => Err(Diagnostic::error(&format!(
                "The debug tag on stream {} received an unexpected value",
                mir.stream(sr).name()
            ))
            .add_span_with_label(
                mir.stream(sr).tags_span()["debug"],
                Some("Found debug tag here"),
                true,
            )
            .into()),
            #[cfg(not(feature = "spanned"))]
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
