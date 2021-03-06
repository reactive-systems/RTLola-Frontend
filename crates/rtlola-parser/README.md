# RTLola Parser
[![Crate](https://img.shields.io/crates/v/rtlola-parser.svg)](https://crates.io/crates/rtlola-parser)
[![API](https://docs.rs/rtlola-parser/badge.svg)](https://docs.rs/rtlola-parser)
[![License](https://img.shields.io/crates/l/rtlola-parser)](https://crates.io/crates/rtlola-parser)

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [rtlola-frontend](https://crates.io/crates/rtlola-frontend) 
* The RTLola high-level intermediate representation including a strong static analysis: [rtlola-hir](https://crates.io/crates/rtlola-hir)
* The RTLola error reporting: [rtlola-reporting](https://crates.io/crates/rtlola-reporting)
* Procedural macros: [rtlola-macros](https://crates.io/crates/rtlola-macros)

# Copyright

Copyright (C) CISPA - Helmholtz Center for Information Security 2021.  Authors: Florian Kohn, Stefan Oswald, Malte Schledjewski, Maximilian Schwenger.
Based on original work at Universität des Saarlandes (C) 2020.  Authors: Jan Baumeister, Florian Kohn, Malte Schledjewski, Maximilian Schwenger, Marvin Stenger, and Leander Tentrup.


