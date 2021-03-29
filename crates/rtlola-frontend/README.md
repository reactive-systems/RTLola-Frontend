# RTLola Frontend

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate summarizes the entire RTLola front-end, which includes several sub-modules:
* A parser for RTLola specifications: [rtlola-parser](crates.io/crates/rtlola-parser) 
* The RTLola high-level intermediate representation including a strong static analysis: [rtlola-hir](crates.io/crates/rtlola-hir)
* The RTLola error reporting: [rtlola-reporting](crates.io/crates/rtlola-reporting)
* Procedural macros: [rtlola-macros](crates.io/crates/rtlola-macros)

# Copyright

Copyright (C) CISPA - Helmholtz Center for Information Security 2021.  Authors: Jan Baumeister, Florian Kohn, Stefan Oswald, Maximilian Schwenger.
Based on original work at Universität des Saarlandes (C) 2020.  Authors: Jan Baumeister, Florian Kohn, Malte Schledjewski, Maximilian Schwenger, Marvin Stenger, and Leander Tentrup.