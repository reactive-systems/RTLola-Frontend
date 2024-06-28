# RTLola Macros
[![Crate](https://img.shields.io/crates/v/rtlola-macros.svg)](https://crates.io/crates/rtlola-macros)
[![API](https://docs.rs/rtlola-macros/badge.svg)](https://docs.rs/rtlola-macros)
[![License](https://img.shields.io/crates/l/rtlola-macros)](https://crates.io/crates/rtlola-macros)

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [rtlola-frontend](https://crates.io/crates/rtlola-frontend) 
* A parser for RTLola specifications: [rtlola-parser](https://crates.io/crates/rtlola-parser) 
* The RTLola high-level intermediate representation including a strong static analysis: [rtlola-hir](https://crates.io/crates/rtlola-hir)
* The RTLola error reporting: [rtlola-reporting](https://crates.io/crates/rtlola-reporting)

# Copyright

Copyright (c) CISPA - Helmholtz Center for Information Security 2021-24.  Author: Maximilian Schwenger.

Procedural macros for more convenient handling of HirModes in the RTLola-Frontend.
