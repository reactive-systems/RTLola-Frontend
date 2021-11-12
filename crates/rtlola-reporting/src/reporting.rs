//! This module contains helper to report messages (warnings/errors)
use std::fmt::Debug;
use std::iter::FromIterator;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::RwLock;

use codespan_reporting::diagnostic::{Diagnostic as RawDiagnostic, Label, Severity};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream, WriteColor};
use codespan_reporting::term::Config;
use serde::{Deserialize, Serialize};

/// Represents a location in the source
// Todo: Change Indirect to Indirect { start: usize, end: usize } to make Span copy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Span {
    /// Direct code reference through byte offset
    Direct {
        /// The start of the span in characters absolute to the beginning of the specification.
        start: usize,
        /// The end of the span in characters absolute to the beginning of the specification.
        end: usize,
    },
    /// Indirect code reference created through ast refactoring
    Indirect(Box<Self>),
    /// An unknown code reference
    Unknown,
}
impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span::Direct {
            start: span.start(),
            end: span.end(),
        }
    }
}

impl From<Span> for Range<usize> {
    fn from(s: Span) -> Range<usize> {
        let (s, e) = s.get_bounds();
        Range { start: s, end: e }
    }
}

impl Span {
    /// Return true if the span is indirect.
    pub fn is_indirect(&self) -> bool {
        match self {
            Span::Direct { .. } => false,
            Span::Indirect(_) => true,
            Span::Unknown => false,
        }
    }

    /// Returns true if the span is unknown.
    pub fn is_unknown(&self) -> bool {
        match self {
            Span::Direct { .. } => false,
            Span::Indirect(_) => false,
            Span::Unknown => true,
        }
    }

    /// Returns the start and end position of the span.
    /// Note: If the span is unknown returns (usize::min, usize::max)
    fn get_bounds(&self) -> (usize, usize) {
        match self {
            Span::Direct { start: s, end: e } => (*s, *e),
            Span::Indirect(s) => s.get_bounds(),
            Span::Unknown => (usize::min_value(), usize::max_value()),
        }
    }

    /// Combines two spans to their union
    pub fn union(&self, other: &Self) -> Self {
        if self.is_unknown() {
            return other.clone();
        }
        if other.is_unknown() {
            return self.clone();
        }
        let (start1, end1) = self.get_bounds();
        let (start2, end2) = other.get_bounds();
        if self.is_indirect() || other.is_indirect() {
            Span::Indirect(Box::new(Span::Direct {
                start: start1.min(start2),
                end: end1.max(end2),
            }))
        } else {
            Span::Direct {
                start: start1.min(start2),
                end: end1.max(end2),
            }
        }
    }
}

/// A handler is responsible for emitting warnings and errors
pub struct Handler {
    /// The number of errors that have already occurred
    error_count: RwLock<usize>,
    /// The number of warnings that have already occurred
    warning_count: RwLock<usize>,
    /// The input file the handler refers to given by a path and its content
    input: SimpleFile<String, String>,
    /// The output the handler is emitting to
    output: RwLock<Box<dyn WriteColor>>,
    /// The config for the error formatting
    config: Config,
}
impl Debug for Handler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handler")
            .field("error_count", &self.error_count)
            .field("warning_count", &self.warning_count)
            .field("input", &self.input)
            .field("config", &self.config)
            .finish()
    }
}

impl Handler {
    /// Creates a new Handler
    /// `input_path` refers to the path of the input file
    /// `input_content` refers to the content of the input file
    pub fn new(input_path: PathBuf, input_content: String) -> Self {
        Handler {
            error_count: RwLock::new(0),
            warning_count: RwLock::new(0),
            input: SimpleFile::new(input_path.to_str().unwrap_or("unknown file").into(), input_content),
            output: RwLock::new(Box::new(StandardStream::stderr(ColorChoice::Always))),
            config: Config::default(),
        }
    }

    /// Creates a new handler without a path.
    pub fn without_file(input_content: String) -> Self {
        Handler {
            error_count: RwLock::new(0),
            warning_count: RwLock::new(0),
            input: SimpleFile::new("unknown file".into(), input_content),
            output: RwLock::new(Box::new(StandardStream::stderr(ColorChoice::Always))),
            config: Config::default(),
        }
    }

    /// Emits a single [Diagnostic] to the terminal
    pub fn emit(&self, diag: &Diagnostic) {
        let mut diag = diag.clone();
        if diag.has_indirect_span {
            diag.inner
                .notes
                .push("Warning was caused indirectly by transformations.".into());
        }
        self.emit_raw(diag.inner);
    }

    /// Emits a [RtLolaError] to the console
    pub fn emit_error(&self, err: &RtLolaError) {
        err.iter().for_each(|diag| self.emit(diag));
    }

    fn emit_raw(&self, diag: RawDiagnostic<()>) {
        match diag.severity {
            Severity::Error => *self.error_count.write().unwrap() += 1,
            Severity::Warning => *self.warning_count.write().unwrap() += 1,
            _ => {},
        }
        term::emit(
            (*self.output.write().unwrap()).as_mut(),
            &self.config,
            &self.input,
            &diag,
        )
        .expect("Could not write diagnostic.");
    }

    /// Returns true if an error has occurred
    pub fn contains_error(&self) -> bool {
        self.emitted_errors() > 0
    }

    /// Returns the number of emitted errors
    pub fn emitted_errors(&self) -> usize {
        *self.error_count.read().unwrap()
    }

    /// Returns the number of emitted warnings
    pub fn emitted_warnings(&self) -> usize {
        *self.warning_count.read().unwrap()
    }

    /// Emits a simple warning with a message
    pub fn warn(&self, message: &str) {
        self.emit_raw(RawDiagnostic::warning().with_message(message))
    }

    /// Emits a warning referring to the code span `span` with and optional label `span_label`
    /// that is printed next to the code fragment
    pub fn warn_with_span(&self, message: &str, span: Span, span_label: Option<&str>) {
        let mut diag = RawDiagnostic::warning().with_message(message);
        if !span.is_unknown() {
            let mut label = Label::primary((), span.clone());
            if let Some(l) = span_label {
                label.message = l.into();
            }
            diag.labels = vec![label];
        }
        if span.is_indirect() {
            diag.notes = vec!["Warning was caused indirectly by transformations.".into()];
        }
        self.emit_raw(diag)
    }

    /// Emits a simple error with a message
    pub fn error(&self, message: &str) {
        self.emit_raw(RawDiagnostic::error().with_message(message))
    }

    /// Emits an error referring to the code span `span` with and optional label `span_label`
    /// that is printed next to the code fragment
    pub fn error_with_span(&self, message: &str, span: Span, span_label: Option<&str>) {
        let mut diag = RawDiagnostic::error().with_message(message);
        if !span.is_unknown() {
            let mut label = Label::primary((), span.clone());
            if let Some(l) = span_label {
                label.message = l.into();
            }
            diag.labels = vec![label];
        }
        if span.is_indirect() {
            diag.notes = vec!["Error was caused indirectly by transformations.".into()];
        }
        self.emit_raw(diag)
    }
}

/// A `Diagnostic` is more flexible way to build and output errors and warnings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// The internal representation of the diagnostic
    pub(crate) inner: RawDiagnostic<()>,
    /// True if the diagnostic refers to at least one indirect span
    pub(crate) has_indirect_span: bool,
}

impl Diagnostic {
    /// Creates a new warning with the message `message`
    pub fn warning(message: &str) -> Self {
        Diagnostic {
            inner: RawDiagnostic::warning().with_message(message),
            has_indirect_span: false,
        }
    }

    /// Creates a new error with the message `message`
    pub fn error(message: &str) -> Self {
        Diagnostic {
            inner: RawDiagnostic::error().with_message(message),
            has_indirect_span: false,
        }
    }

    /// Adds a code span to the diagnostic.
    /// The `label` is printed next to the code fragment the span refers to.
    /// If `primary` is set to true the span is treated as the primary code fragment.
    pub fn add_span_with_label(mut self, span: Span, label: Option<&str>, primary: bool) -> Self {
        if span.is_unknown() {
            return self;
        }
        self.has_indirect_span |= span.is_indirect();
        let mut rep_label = if primary {
            Label::primary((), span)
        } else {
            Label::secondary((), span)
        };
        if let Some(l) = label {
            rep_label.message = l.into();
        }
        self.inner.labels.push(rep_label);
        self
    }

    /// Adds a code span to the diagnostic if the span is available.
    /// The `label` is printed next to the code fragment the span refers to.
    /// If `primary` is set to true the span is treated as the primary code fragment.
    pub fn maybe_add_span_with_label(mut self, span: Option<Span>, label: Option<&str>, primary: bool) -> Self {
        let span = match span {
            None | Some(Span::Unknown) => return self,
            Some(s) => s,
        };
        self.has_indirect_span |= span.is_indirect();
        let mut rep_label = if primary {
            Label::primary((), span)
        } else {
            Label::secondary((), span)
        };
        if let Some(l) = label {
            rep_label.message = l.into();
        }
        self.inner.labels.push(rep_label);
        self
    }

    /// Adds a note to the bottom of the diagnostic.
    pub fn add_note(mut self, note: &str) -> Self {
        self.inner.notes.push(note.into());
        self
    }
}

impl From<Diagnostic> for RawDiagnostic<()> {
    fn from(diag: Diagnostic) -> Self {
        diag.inner
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// An error type to collect diagnostics throughout the frontend.
pub struct RtLolaError {
    errors: Vec<Diagnostic>,
}

impl RtLolaError {
    /// Creates a new error
    pub fn new() -> Self {
        RtLolaError { errors: vec![] }
    }

    /// Adds a [Diagnostic] to the error
    pub fn add(&mut self, diag: Diagnostic) {
        self.errors.push(diag)
    }

    /// Returns a slice of all [Diagnostic]s of the error
    pub fn as_slice(&self) -> &[Diagnostic] {
        self.errors.as_slice()
    }

    /// Returns the number of Diagnostics with the severity error
    pub fn num_errors(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| matches!(e.inner.severity, Severity::Error))
            .count()
    }

    /// Returns the number of Diagnostics with the severity warning
    pub fn num_warnings(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| matches!(e.inner.severity, Severity::Warning))
            .count()
    }

    /// Merges to errors into one by combining the internal collections
    pub fn join(&mut self, mut other: RtLolaError) {
        self.errors.append(&mut other.errors)
    }

    /// Returns an iterator over the [Diagnostic]s of the error
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.errors.iter()
    }

    /// Combines to Results with an RtLolaError as the Error type into a single Result.
    /// If both results are Ok then `op` is applied to these values to construct the new Ok value.
    /// If one of the errors is Err then the Err is returned
    /// If both Results are errors then the RtLolaErrors are merged using [RtLolaError::join] and returned.
    pub fn combine<L, R, U, F: FnOnce(L, R) -> U>(
        left: Result<L, RtLolaError>,
        right: Result<R, RtLolaError>,
        op: F,
    ) -> Result<U, RtLolaError> {
        match (left, right) {
            (Ok(l), Ok(r)) => Ok(op(l, r)),
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => Err(e),
            (Err(mut l), Err(r)) => {
                l.join(r);
                Err(l)
            },
        }
    }
}

impl Default for RtLolaError {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for RtLolaError {
    type IntoIter = std::vec::IntoIter<Self::Item>;
    type Item = Diagnostic;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

impl FromIterator<Diagnostic> for RtLolaError {
    fn from_iter<T: IntoIterator<Item = Diagnostic>>(iter: T) -> Self {
        RtLolaError {
            errors: iter.into_iter().collect(),
        }
    }
}

impl From<Diagnostic> for RtLolaError {
    fn from(diag: Diagnostic) -> Self {
        RtLolaError { errors: vec![diag] }
    }
}

impl From<Result<(), RtLolaError>> for RtLolaError {
    fn from(res: Result<(), RtLolaError>) -> Self {
        match res {
            Ok(()) => RtLolaError::new(),
            Err(e) => e,
        }
    }
}

impl From<RtLolaError> for Result<(), RtLolaError> {
    fn from(e: RtLolaError) -> Self {
        if e.errors
            .iter()
            .filter(|e| matches!(e.inner.severity, Severity::Error))
            .count()
            == 0
        {
            Ok(())
        } else {
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_span() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.error_with_span("Unknown Type", span, Some("here".into()));
        assert_eq!(handler.emitted_errors(), 1);
    }

    #[test]
    fn warning_span() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.warn_with_span("Unknown Type", span, Some("here".into()));
        assert_eq!(handler.emitted_warnings(), 1);
    }

    #[test]
    fn error() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        handler.error("Unknown Type");
        assert_eq!(handler.emitted_errors(), 1);
    }

    #[test]
    fn warning() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        handler.warn("Unknown Type");
        assert_eq!(handler.emitted_warnings(), 1);
    }

    #[test]
    fn error_span_no_label() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.error_with_span("Unknown Type", span, None);
        assert_eq!(handler.emitted_errors(), 1);
    }

    #[test]
    fn custom() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span1 = Span::Direct { start: 9, end: 12 };
        let span2 = Span::Indirect(Box::new(Span::Direct { start: 20, end: 21 }));
        let span3 = Span::Direct { start: 24, end: 25 };
        handler.emit(
            &Diagnostic::error("Failed with love")
                .add_span_with_label(span1, Some("here"), true)
                .add_span_with_label(span2, Some("and here"), false)
                .maybe_add_span_with_label(None, Some("Maybe there is no span"), false)
                .maybe_add_span_with_label(Some(span3), None, false)
                .add_note("This is a note"),
        );
        assert_eq!(handler.emitted_errors(), 1);
    }
}
