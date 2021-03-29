# RTLola High-level Intermediate Representation

RTLola is a stream-based runtime verification framework.  It parses an RTLola specification, analyses it, and generates executable monitors for it.
The framework is separated into a front-end and several back-ends.

This crate is part of the RTLola front-end, which includes several sub-modules:
* Main Crate: The RTLola front-end: [crates.io/crates/rtlola-frontend](rtlola-frontend) 
* A parser for RTLola specifications: [crates.io/crates/rtlola-parser](rtlola-parser) 
* The RTLola error reporting: [crates.io/crates/rtlola-reporting](rtlola-reporting)
* Procedural macros: [crates.io/crates/rtlola-macros](rtlola-macros)

# Copyright
Copyright (C) CISPA - Helmholtz Center for Information Security 2021.  Authors: Jan Baumeister, Florian Kohn, Stefan Oswald, Maximilian Schwenger.
Based on original work at Universität des Saarlandes (C) 2020.  Authors: Jan Baumeister, Florian Kohn, Malte Schledjewski, Maximilian Schwenger, Marvin Stenger, and Leander Tentrup.