extern crate rusttyc;
extern crate streamlab_frontend as front;

mod pacing_ast_climber;
mod pacing_types;
mod rtltc;
mod value_ast_climber;
mod value_types;

use crate::rtltc::LolaTypChecker;
use front::ast::LolaSpec;
use front::parse::SourceMapper;
use front::reporting::Handler;
use front::FrontendConfig;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::option::Option;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Require at least one input specification file!")
    } else {
        let input: String = args[1].to_owned();

        let path = Path::new(&input);
        let display = path.display();
        let mut file = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}", display, why.description()),
            Ok(f) => f,
        };
        let mut prog = String::new();
        match file.read_to_string(&mut prog) {
            Err(why) => panic!("couldn't read {}: {}", display, why.description()),
            Ok(_) => (),
        }

        let mut s = "".to_string();
        for c in input.chars() {
            if c == '/' {
                s = "".to_string();
            } else {
                s.push(c);
            }
        }

        let name = s.clone();
        let p = prog.clone();
        let spec = &p;
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = front::parse::parse(spec, &handler, front::FrontendConfig::default());

        //let ir = front::parse(&name, &p, front::FrontendConfig::default());
        let lola_spec = match ast {
            Err(why) => panic!("parsing error: {}", why), //TODO
            Ok(parsed_spec) => parsed_spec,
        };

        let mut na =
            front::analysis::naming::NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table: front::analysis::naming::DeclarationTable = na.check(&lola_spec);

        let checker = LolaTypChecker::new(&lola_spec, decl_table);

        print!("{:#?}", checker.generate_raw_table());
    }
}
