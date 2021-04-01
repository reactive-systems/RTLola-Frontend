# RTLola Macros

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [rtlola-frontend](https://crates.io/crates/rtlola-frontend) 
* A parser for RTLola specifications: [rtlola-parser](https://crates.io/crates/rtlola-parser) 
* The RTLola high-level intermediate representation including a strong static analysis: [rtlola-hir](https://crates.io/crates/rtlola-hir)
* The RTLola error reporting: [rtlola-reporting](https://crates.io/crates/rtlola-reporting)

# Copyright

Copyright (c) CISPA - Helmholtz Center for Information Security 2021.  Author: Maximilian Schwenger.

Procedural macros for more convenient handling of HirModes in the RTLola-Frontend.
