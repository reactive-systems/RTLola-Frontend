use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

use rtlola_parser::parse;
use rtlola_reporting::RtLolaError;
use serde::Serialize;

use crate::{config::FrontendConfig, fully_analyzed};

#[derive(Debug, Clone, Copy)]
pub(crate) struct AnalysisTracer {
    parse_start: Option<Instant>,
    parse_end: Option<Instant>,
    analysis_start: Option<Instant>,
    analysis_end: Option<Instant>,
    privacy_start: Option<Instant>,
    privacy_end: Option<Instant>,
    privacy_heuristic_start: Option<Instant>,
    privacy_heuristic_end: Option<Instant>,
}

pub(crate) static BENCHMARK_TRACER: Mutex<AnalysisTracer> = Mutex::new(AnalysisTracer::new());

impl AnalysisTracer {
    const fn new() -> Self {
        AnalysisTracer {
            parse_start: None,
            parse_end: None,
            analysis_start: None,
            analysis_end: None,
            privacy_start: None,
            privacy_end: None,
            privacy_heuristic_start: None,
            privacy_heuristic_end: None,
        }
    }

    pub(crate) fn start_parse(&mut self) {
        self.parse_start.replace(Instant::now());
    }

    pub(crate) fn end_parse(&mut self) {
        self.parse_end.replace(Instant::now());
    }

    pub(crate) fn start_analysis(&mut self) {
        self.analysis_start.replace(Instant::now());
    }

    pub(crate) fn end_analysis(&mut self) {
        self.analysis_end.replace(Instant::now());
    }

    pub(crate) fn start_privacy_analysis(&mut self) {
        self.privacy_start.replace(Instant::now());
    }

    pub(crate) fn end_privacy_analysis(&mut self) {
        self.privacy_end.replace(Instant::now());
    }

    pub(crate) fn start_privacy_heuristic(&mut self) {
        self.privacy_heuristic_start.replace(Instant::now());
    }

    pub(crate) fn end_privacy_heuristic(&mut self) {
        self.privacy_heuristic_end.replace(Instant::now());
    }

    fn parse_duration(&self) -> Duration {
        self.parse_end
            .unwrap()
            .duration_since(self.parse_start.unwrap())
    }

    fn analysis_duration(&self) -> Duration {
        self.analysis_end
            .unwrap()
            .duration_since(self.analysis_start.unwrap())
    }

    fn privacy_duration(&self) -> Duration {
        self.privacy_end
            .unwrap()
            .duration_since(self.privacy_start.unwrap())
    }

    fn privacy_heuristic_duration(&self) -> Option<Duration> {
        self.privacy_heuristic_start
            .map(|start| start.duration_since(self.privacy_heuristic_end.unwrap()))
    }

    fn reset(&mut self) {
        *self = Self::new()
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchmarkResults {
    #[serde(serialize_with = "duration_secs")]
    parse_duration: Duration,
    #[serde(serialize_with = "duration_secs")]
    analysis_duration: Duration,
    #[serde(serialize_with = "duration_secs")]
    privacy_duration: Duration,
    #[serde(serialize_with = "optional_duration")]
    heuristic_duration: Option<Duration>,
}

fn duration_secs<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_f64(d.as_secs_f64())
}

fn optional_duration<S>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match d {
        Some(d) => duration_secs(d, s),
        None => s.serialize_str(""),
    }
}

pub fn benchmark(config: &FrontendConfig) -> Result<BenchmarkResults, RtLolaError> {
    let mut tracer = BENCHMARK_TRACER.lock().unwrap();
    tracer.reset();
    tracer.start_parse();
    let ast = parse(config.parser_config())?;
    tracer.end_parse();
    tracer.start_analysis();
    drop(tracer);
    let hir = fully_analyzed(ast, &config)?;
    std::hint::black_box(hir);
    let mut tracer = BENCHMARK_TRACER.lock().unwrap();
    tracer.end_analysis();
    Ok({
        BenchmarkResults {
            parse_duration: tracer.parse_duration(),
            analysis_duration: tracer.analysis_duration(),
            privacy_duration: tracer.privacy_duration(),
            heuristic_duration: tracer.privacy_heuristic_duration(),
        }
    })
}
