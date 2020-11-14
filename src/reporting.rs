//! This module contains helper to report messages (warnings/errors)
use codespan_reporting::diagnostic::{Diagnostic as RepDiagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream, WriteColor};
use codespan_reporting::term::Config;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::ops::Range;
use std::path::PathBuf;
use uom::lib::sync::RwLock;

/// Represents a location in the source
#[derive(Debug, Clone)]
pub enum Span {
    /// Direct code reference through byte offset
    Direct { start: usize, end: usize },
    /// Indirect code reference created through ast refactoring
    Indirect(Box<Self>),
}
impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span::Direct { start: span.start(), end: span.end() }
    }
}
impl Into<Range<usize>> for Span {
    fn into(self) -> Range<usize> {
        match self {
            Span::Direct { start: s, end: e } => Range { start: s, end: e },
            Span::Indirect(span) => (*span).into(),
        }
    }
}
impl Span {
    pub fn is_indirect(&self) -> bool {
        match self {
            Span::Direct { .. } => false,
            Span::Indirect(_) => true,
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

impl Handler {
    pub fn new(input_path: PathBuf, input_content: String) -> Self {
        Handler {
            error_count: RwLock::new(0),
            warning_count: RwLock::new(0),
            input: SimpleFile::new(
                input_path.file_name().unwrap_or(OsStr::new("unknown file")).to_str().unwrap_or("unknown file").into(),
                input_content,
            ),
            output: RwLock::new(Box::new(StandardStream::stderr(ColorChoice::Always))),
            config: Config::default(),
        }
    }

    pub fn emit(&self, diag: &RepDiagnostic<()>) {
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
        let mut label = Label::primary((), span.clone());
        if let Some(l) = span_label {
            label.message = l.into();
        }
        diag.labels = vec![label];
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
        let mut label = Label::primary((), span.clone());
        if let Some(l) = span_label {
            label.message = l.into();
        }
        diag.labels = vec![label];
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
    fn warning(handler: &'a Handler, message: &str) -> Self {
        Diagnostic {
            handler,
            diag: RepDiagnostic::warning().with_message(message),
            emitted: false,
            has_indirect_span: false,
            indirect_note_text: "Warning was caused indirectly by transformations.".into(),
        }
    }

    fn error(handler: &'a Handler, message: &str) -> Self {
        Diagnostic {
            handler,
            diag: RepDiagnostic::error().with_message(message),
            emitted: false,
            has_indirect_span: false,
            indirect_note_text: "Error was caused indirectly by transformations.".into(),
        }
    }

    pub fn emit(mut self) {
        if self.has_indirect_span {
            self.diag.notes.push(self.indirect_note_text);
        }
        if !self.emitted {
            self.handler.emit(&self.diag);
        }
        self.emitted = true;
    }

    pub fn add_span_with_label(mut self, span: Span, label: &str, primary: bool) -> Self {
        self.has_indirect_span = self.has_indirect_span || span.is_indirect();
        let label = if primary {
            Label::primary((), span).with_message(label)
        } else {
            Label::secondary((), span).with_message(label)
        };
        self.diag.labels.push(label);
        self
    }

    pub fn add_note(mut self, note: &str) -> Self {
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
    }

    #[test]
    fn warning_span() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.warn_with_span("Unknown Type", span, Some("here".into()));
    }

    #[test]
    fn error() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.error("Unknown Type");
    }

    #[test]
    fn warning() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.warn("Unknown Type");
    }

    #[test]
    fn error_span_no_label() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span = Span::Direct { start: 9, end: 12 };
        handler.error_with_span("Unknown Type", span, None);
    }

    #[test]
    fn custom() {
        let handler = Handler::new(PathBuf::from("stdin"), "input i: Int\noutput x = 5".into());
        let span1 = Span::Direct { start: 9, end: 12 };
        let span2 = Span::Indirect(Box::new(Span::Direct { start: 20, end: 21 }));
        Diagnostic::error(&handler, "Failed with love")
            .add_span_with_label(span1, "here", true)
            .add_span_with_label(span2, "and here", false)
            .add_note("This is a note")
            .emit();
    }
}
