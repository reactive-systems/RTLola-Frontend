# RTLola Reporting
[![Crate](https://img.shields.io/crates/v/rtlola-reporting.svg)](https://crates.io/crates/rtlola-reporting)
[![API](https://docs.rs/rtlola-reporting/badge.svg)](https://docs.rs/rtlola-reporting)
[![License](https://img.shields.io/crates/l/rtlola-reporting)](https://crates.io/crates/rtlola-reporting)

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [rtlola-frontend](https://crates.io/crates/rtlola-frontend) 
* A parser for RTLola specifications: [rtlola-parser](https://crates.io/crates/rtlola-parser) 
* The RTLola high-level intermediate representation including a strong static analysis: [rtlola-hir](https://crates.io/crates/rtlola-hir)
* Procedural macros: [rtlola-macros](https://crates.io/crates/rtlola-macros)

# Copyright

Copyright (C) CISPA - Helmholtz Center for Information Security 2021.  Authors: Florian Kohn, Maximilian Schwenger.
Based on original work at Universität des Saarlandes (C) 2020.  Authors: Jan Baumeister, Florian Kohn, Malte Schledjewski, Maximilian Schwenger, Marvin Stenger, and Leander Tentrup.
