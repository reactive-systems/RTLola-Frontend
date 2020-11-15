//! This module contains helper to report messages (warnings/errors)
use codespan_reporting::diagnostic::{Diagnostic as RepDiagnostic, Label, Severity};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream, WriteColor};
use codespan_reporting::term::Config;
use std::fmt::Debug;
use std::ops::Range;
use std::path::PathBuf;
use uom::lib::sync::RwLock;

/// Represents a location in the source
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Span {
    /// Direct code reference through byte offset
    Direct { start: usize, end: usize },
    /// Indirect code reference created through ast refactoring
    Indirect(Box<Self>),
    /// An unknown code reference
    Unknown,
}
impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span::Direct { start: span.start(), end: span.end() }
    }
}
impl Into<Range<usize>> for Span {
    fn into(self) -> Range<usize> {
        let (s, e) = self.get_bounds();
        Range { start: s, end: e }
    }
}
impl Span {
    pub fn is_indirect(&self) -> bool {
        match self {
            Span::Direct { .. } => false,
            Span::Indirect(_) => true,
            Span::Unknown => false,
        }
    }
    pub fn is_unknown(&self) -> bool {
        match self {
            Span::Direct { .. } => false,
            Span::Indirect(_) => false,
            Span::Unknown => true,
        }
    }

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
            Span::Indirect(Box::new(Span::Direct { start: start1.min(start2), end: end1.max(end2) }))
        } else {
            Span::Direct { start: start1.min(start2), end: end1.max(end2) }
        }
    }
}

/// A handler is responsible for emitting warnings and errors
pub struct Handler {
    error_count: RwLock<usize>,
    warning_count: RwLock<usize>,
    input: SimpleFile<String, String>,
    output: RwLock<Box<dyn WriteColor>>,
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
    pub fn new(input_path: PathBuf, input_content: String) -> Self {
        Handler {
            error_count: RwLock::new(0),
            warning_count: RwLock::new(0),
            input: SimpleFile::new(input_path.to_str().unwrap_or("unknown file").into(), input_content),
            output: RwLock::new(Box::new(StandardStream::stderr(ColorChoice::Always))),
            config: Config::default(),
        }
    }

    pub fn emit(&self, diag: &RepDiagnostic<()>) {
        match diag.severity {
            Severity::Error => *self.error_count.write().unwrap() += 1,
            Severity::Warning => *self.warning_count.write().unwrap() += 1,
            _ => {}
        }
        //#[cfg(not(test))]
        term::emit((*self.output.write().unwrap()).as_mut(), &self.config, &self.input, diag)
            .expect("Could not write diagnostic.");
    }

    pub fn contains_error(&self) -> bool {
        self.emitted_errors() > 0
    }

    pub fn emitted_errors(&self) -> usize {
        *self.error_count.read().unwrap()
    }

    pub fn emitted_warnings(&self) -> usize {
        *self.warning_count.read().unwrap()
    }

    pub fn warn(&self, message: &str) {
        self.emit(&RepDiagnostic::warning().with_message(message))
    }

    pub fn warn_with_span(&self, message: &str, span: Span, span_label: Option<&str>) {
        let mut diag = RepDiagnostic::warning().with_message(message);
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
        self.emit(&diag)
    }

    pub fn error(&self, message: &str) {
        self.emit(&RepDiagnostic::error().with_message(message))
    }

    pub fn error_with_span(&self, message: &str, span: Span, span_label: Option<&str>) {
        let mut diag = RepDiagnostic::error().with_message(message);
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
        self.emit(&diag)
    }
}

/// `Diagnostic` a more flexible way to build a diagnostic.
pub struct Diagnostic<'a> {
    handler: &'a Handler,
    diag: RepDiagnostic<()>,
    emitted: bool,
    has_indirect_span: bool,
    indirect_note_text: String,
}

impl<'a> Diagnostic<'a> {
    #[allow(dead_code)]
    pub(crate) fn warning(handler: &'a Handler, message: &str) -> Self {
        Diagnostic {
            handler,
            diag: RepDiagnostic::warning().with_message(message),
            emitted: false,
            has_indirect_span: false,
            indirect_note_text: "Warning was caused indirectly by transformations.".into(),
        }
    }

    pub(crate) fn error(handler: &'a Handler, message: &str) -> Self {
        Diagnostic {
            handler,
            diag: RepDiagnostic::error().with_message(message),
            emitted: false,
            has_indirect_span: false,
            indirect_note_text: "Error was caused indirectly by transformations.".into(),
        }
    }

    pub(crate) fn emit(mut self) {
        assert!(!self.emitted, "Diagnostic can only be emitted once!");
        if self.has_indirect_span {
            self.diag.notes.push(self.indirect_note_text);
        }
        self.handler.emit(&self.diag);
        self.emitted = true;
    }

    pub(crate) fn add_span_with_label(mut self, span: Span, label: Option<&str>, primary: bool) -> Self {
        if span.is_unknown() {
            return self;
        }
        self.has_indirect_span |= span.is_indirect();
        let mut rep_label = if primary { Label::primary((), span) } else { Label::secondary((), span) };
        if let Some(l) = label {
            rep_label.message = l.into();
        }
        self.diag.labels.push(rep_label);
        self
    }
    #[allow(dead_code)]
    pub(crate) fn add_note(mut self, note: &str) -> Self {
        self.diag.notes.push(note.into());
        self
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
        Diagnostic::error(&handler, "Failed with love")
            .add_span_with_label(span1, Some("here"), true)
            .add_span_with_label(span2, Some("and here"), false)
            .add_note("This is a note")
            .emit();
        assert_eq!(handler.emitted_errors(), 1);
    }
}
