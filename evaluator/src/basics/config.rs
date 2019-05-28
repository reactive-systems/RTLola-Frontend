use super::{InputSource, OutputChannel};

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub source: InputSource,
    pub verbosity: Verbosity,
    pub output_channel: OutputChannel,
    pub evaluator: EvaluatorChoice,
    pub mode: ExecutionMode,
    pub time_presentation: TimeRepresentation,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Verbosity {
    /// Suppresses any kind of logging.
    Silent,
    /// Prints statistical information like number of events, triggers, etc.
    Progress,
    /// Prints nothing but runtime warnings about potentially critical states, e.g. dropped
    /// evaluation cycles.
    WarningsOnly,
    /// Prints only triggers and runtime warnings.
    Triggers,
    /// Prints information about all or a subset of output streams whenever they produce a new
    /// value.
    Outputs,
    /// Prints fine-grained debug information. Not suitable for production.
    Debug,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecutionMode {
    Offline,
    Online,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvaluatorChoice {
    ClosureBased,
    Interpreted,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeRepresentation {
    Hide,
    Relative(TimeFormat),
    Absolute(TimeFormat),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeFormat {
    UIntNanos,
    FloatSecs,
    HumanTime,
}

impl EvalConfig {
    pub fn new(
        source: InputSource,
        verbosity: Verbosity,
        output: OutputChannel,
        evaluator: EvaluatorChoice,
        mode: ExecutionMode,
        time_presentation: TimeRepresentation,
    ) -> Self {
        EvalConfig { source, verbosity, output_channel: output, evaluator, mode, time_presentation }
    }

    pub fn debug() -> Self {
        let mut cfg = EvalConfig::default();
        cfg.verbosity = Verbosity::Debug;
        cfg
    }

    pub fn release(
        path: String,
        output: OutputChannel,
        evaluator: EvaluatorChoice,
        mode: ExecutionMode,
        time_presentation: TimeRepresentation,
    ) -> Self {
        EvalConfig::new(
            InputSource::file(path, None, None),
            Verbosity::Triggers,
            output,
            evaluator,
            mode,
            time_presentation,
        )
    }
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            source: InputSource::StdIn,
            verbosity: Verbosity::Triggers,
            output_channel: OutputChannel::StdOut,
            evaluator: EvaluatorChoice::ClosureBased,
            mode: ExecutionMode::Offline,
            time_presentation: TimeRepresentation::Hide,
        }
    }
}