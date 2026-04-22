use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use rtlola_frontend::{parse, FrontendConfig, Handler, ParserConfigExt, PrivacyHeuristic};
use rtlola_parser::ParserConfig;

#[derive(Parser)]
struct Args {
    /// The specification to parse and analyse
    spec: PathBuf,
    /// Add privacy barriers with the given privacy parameter
    /// Omitting this argument does not add any privacy
    #[clap(long)]
    privacy_parameter: Option<f64>,
    /// Use the given heuristic to choose barrier positions
    #[clap(long,requires="privacy_parameter", value_enum, default_value_t=HeuristicArg::Inputs)]
    privacy_heuristic: HeuristicArg,
}

#[derive(ValueEnum, Clone, Debug, Copy)]
enum HeuristicArg {
    /// Add barriers directly at the input level
    Inputs,
    /// Add barriers as close to the public outputs as possible
    Deep,
    /// Add the least number of privacy barriers
    LeastCutpoints,
}

impl From<HeuristicArg> for PrivacyHeuristic {
    fn from(value: HeuristicArg) -> Self {
        match value {
            HeuristicArg::Inputs => PrivacyHeuristic::Inputs,
            HeuristicArg::Deep => PrivacyHeuristic::Deep,
            HeuristicArg::LeastCutpoints => PrivacyHeuristic::LeastCutpoints,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let Args {
        spec,
        privacy_parameter,
        privacy_heuristic,
    } = Args::parse();
    let pconfig = ParserConfig::from_path(spec).context("error loading specification file")?;
    let config = if let Some(privacy_parameter) = privacy_parameter {
        pconfig
            .with_privacy_parameter(privacy_parameter)
            .with_privacy_heuristic(privacy_heuristic.into())
    } else {
        FrontendConfig::from(&pconfig)
    };
    let mir = match parse(config) {
        Ok(mir) => mir,
        Err(e) => {
            let handler = Handler::from(&pconfig);
            handler.emit_error(&e);
            anyhow::bail!("Error analyzing specification");
        }
    };
    println!("{mir}");
    Ok(())
}
