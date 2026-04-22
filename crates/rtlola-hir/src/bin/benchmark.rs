use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rtlola_hir::{
    benchmark::benchmark,
    config::{ParserConfigExt, PrivacyHeuristic},
};
use rtlola_parser::ParserConfig;

#[derive(Parser, Debug, Clone)]
struct Args {
    spec: PathBuf,
    #[clap(long, value_enum)]
    heuristic: HeuristicArg,
    #[clap(short, long)]
    runs: u64,
    #[clap(long, default_value_t = 1.0)]
    parameter: f64,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum HeuristicArg {
    Inputs,
    Deep,
    LeastCutpoints,
    NoPrivate,
}

fn main() {
    let Args {
        spec,
        heuristic,
        runs,
        parameter,
    } = Args::parse();
    let config = ParserConfig::from_path(spec).unwrap();
    let config = if heuristic != HeuristicArg::NoPrivate {
        let heuristic = match heuristic {
            HeuristicArg::Inputs => PrivacyHeuristic::Inputs,
            HeuristicArg::Deep => PrivacyHeuristic::Deep,
            HeuristicArg::LeastCutpoints => PrivacyHeuristic::LeastCutpoints,
            HeuristicArg::NoPrivate => unreachable!(),
        };
        config
            .with_privacy_parameter(parameter)
            .with_privacy_heuristic(heuristic)
    } else {
        (&config).into()
    };

    let mut csv = csv::Writer::from_writer(std::io::stdout());

    for _ in 0..runs {
        let results = benchmark(&config).unwrap();
        csv.serialize(results).unwrap();
    }
}
