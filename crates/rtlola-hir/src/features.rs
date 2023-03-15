use std::collections::{HashMap, HashSet};

use rtlola_parser::ast::WindowOperation;
use rtlola_reporting::{Diagnostic, RtLolaError, Span};

use crate::hir::{ConcretePacingType, DiscreteAggr, Feature, Output, SlidingAggr, Window};
use crate::type_check::ConcreteValueType;

#[derive(Debug, Clone)]
/// The RTLola feature of parameterized streams.
pub struct Parameterized {}
impl Feature for Parameterized {
    fn name(&self) -> &'static str {
        "Parameterized"
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        if output.params.is_empty() {
            Ok(())
        } else {
            Err(
                Diagnostic::error("Unsupported Feature: Parameters are not supported by the backend.")
                    .add_span_with_label(output.span, Some("Found parameterized output stream here"), true)
                    .into(),
            )
        }
    }
}

#[derive(Debug, Clone)]
/// The RTLola feature to dynamically create streams.
pub struct Spawned {}
impl Feature for Spawned {
    fn name(&self) -> &'static str {
        "Spawned"
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        if output.spawn.is_none() {
            Ok(())
        } else {
            Err(Diagnostic::error(
                "Unsupported Feature: Dynamically creating streams through spawn is not supported by the backend.",
            )
            .add_span_with_label(output.span, Some("Found spawned stream here"), true)
            .into())
        }
    }
}

#[derive(Debug, Clone)]
/// The RTLola feature to conditionally evaluate streams.
pub struct Filtered {}
impl Feature for Filtered {
    fn name(&self) -> &'static str {
        "Filtered"
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        if output.eval.condition.is_none() {
            Ok(())
        } else {
            Err(Diagnostic::error("Unsupported Feature: Conditionally evaluating a stream through when clauses is not supported by the backend.").add_span_with_label(output.span, Some("Found stream with when clause here."), true).into())
        }
    }
}

#[derive(Debug, Clone)]
/// The RTLola feature to dynamically close streams.
pub struct Closed {}
impl Feature for Closed {
    fn name(&self) -> &'static str {
        "Closed"
    }

    fn exclude_output(&self, output: &Output) -> Result<(), RtLolaError> {
        if output.close.is_none() {
            Ok(())
        } else {
            Err(
                Diagnostic::error("Unsupported Feature: Dynamically closing streams is not supported by the backend.")
                    .add_span_with_label(output.span, Some("Found closed stream here"), true)
                    .into(),
            )
        }
    }
}

#[derive(Debug, Clone, Default)]
/// A generic Feature to exclude a set of sliding [WindowOperation]s.
pub struct SlidingWindows {
    /// The set of unsupported window operations.
    /// An empty set symbolizes that sliding windows are not supported at all.
    unsupported: HashSet<WindowOperation>,
}

impl SlidingWindows {
    /// Creates a new sliding windows feature given a set of unsupported window operations.
    pub fn new(unsupported: HashSet<WindowOperation>) -> Self {
        Self { unsupported }
    }
}

impl Feature for SlidingWindows {
    fn name(&self) -> &'static str {
        "Sliding Windows"
    }

    fn exclude_sliding_window(&self, span: &Span, window: &Window<SlidingAggr>) -> Result<(), RtLolaError> {
        let op = &window.aggr.op;
        if self.unsupported.is_empty() {
            Err(
                Diagnostic::error("Unsupported Feature: Sliding windows are not supported by the backend.")
                    .add_span_with_label(*span, Some("Found sliding window here"), true)
                    .into(),
            )
        } else if self.unsupported.contains(op) {
            Err(Diagnostic::error(&format!(
                "Unsupported Feature: Sliding window operation <{op}> is not supported by the backend."
            ))
            .add_span_with_label(*span, Some("Found sliding window here"), true)
            .into())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Default)]
/// A generic feature to exclude a set of discrete [WindowOperation]s.
pub struct DiscreteWindows {
    /// The set of unsupported window operations.
    /// An empty set symbolizes that sliding windows are not supported at all.
    unsupported: HashSet<WindowOperation>,
}

impl DiscreteWindows {
    /// Creates a new discrete windows feature given a set of unsupported window operations.
    pub fn new(unsupported: HashSet<WindowOperation>) -> Self {
        Self { unsupported }
    }
}

impl Feature for DiscreteWindows {
    fn name(&self) -> &'static str {
        "Discrete Windows"
    }

    fn exclude_discrete_window(&self, span: &Span, window: &Window<DiscreteAggr>) -> Result<(), RtLolaError> {
        let op = &window.aggr.op;
        if self.unsupported.is_empty() {
            Err(
                Diagnostic::error("Unsupported Feature: Discrete windows are not supported by the backend.")
                    .add_span_with_label(*span, Some("Found discrete window here"), true)
                    .into(),
            )
        } else if self.unsupported.contains(op) {
            Err(Diagnostic::error(&format!(
                "Unsupported Feature: Discrete window operation <{op}> is not supported by the backend."
            ))
            .add_span_with_label(*span, Some("Found discrete window here"), true)
            .into())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
// The RTLola feature of periodic evaluation.
pub struct Periodics {}

impl Feature for Periodics {
    fn name(&self) -> &'static str {
        "Periodics"
    }

    fn exclude_pacing_type(&self, span: &Span, ty: &ConcretePacingType) -> Result<(), RtLolaError> {
        match ty {
            ConcretePacingType::FixedPeriodic(_) | ConcretePacingType::Periodic => {
                let str_ty = ty.to_pretty_string(&HashMap::new());
                Err(
                    Diagnostic::error("Unsupported Feature: Periodic evaluation is not supported by the backend.")
                        .add_span_with_label(*span, Some(&format!("Found periodic pacing <{str_ty}> here")), true)
                        .into(),
                )
            },
            ConcretePacingType::Constant | ConcretePacingType::Event(_) => Ok(()),
        }
    }
}

#[derive(Debug, Clone)]
/// A generic feature to exclude different Value Types.
pub struct ValueTypes {
    /// The set of unsupported value types.
    unsupported: HashSet<ConcreteValueType>,
}

impl ValueTypes {
    /// Creates a new Value Type feature given a set of unsupported value types.
    pub fn new(unsupported: HashSet<ConcreteValueType>) -> Self {
        Self { unsupported }
    }
}

impl Feature for ValueTypes {
    fn name(&self) -> &'static str {
        "ValueTypes"
    }

    fn exclude_value_type(&self, span: &Span, ty: &ConcreteValueType) -> Result<(), RtLolaError> {
        if self.unsupported.contains(ty) {
            Err(Diagnostic::error("Unsupported Feature: Value type not supported.")
                .add_span_with_label(*span, Some(&format!("Found unsupported value type <{ty}> here")), true)
                .into())
        } else {
            Ok(())
        }
    }
}
