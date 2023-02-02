# RTLola High-level Intermediate Representation
[![Crate](https://img.shields.io/crates/v/rtlola-hir.svg)](https://crates.io/crates/rtlola-hir)
[![API](https://docs.rs/rtlola-hir/badge.svg)](https://docs.rs/rtlola-hir)
[![License](https://img.shields.io/crates/l/rtlola-hir)](https://crates.io/crates/rtlola-hir)

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [rtlola-frontend](https://crates.io/crates/rtlola-frontend) 
* A parser for RTLola specifications: [rtlola-parser](https://crates.io/crates/rtlola-parser) 
* The RTLola error reporting: [rtlola-reporting](https://crates.io/crates/rtlola-reporting)
* Procedural macros: [rtlola-macros](https://crates.io/crates/rtlola-macros)

# Copyright
Copyright (C) CISPA - Helmholtz Center for Information Security 2021-2023.  Authors: Jan Baumeister, Florian Kohn, Stefan Oswald, Maximilian Schwenger.
Based on original work at Universität des Saarlandes (C) 2020.  Authors: Jan Baumeister, Florian Kohn, Malte Schledjewski, Maximilian Schwenger, Marvin Stenger, and Leander Tentrup.