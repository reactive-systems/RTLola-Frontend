# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - TBD

### Added
- Added `is_eval_filtered` method to stream trait.

## [0.6.1] - 02.02.2023

### Changed
- adapt to new rust version
- update Copyright in `README.md`

## [0.6.0] - 19.12.2022

### Changed
- Include stream access kind in Mir
- Update for new Syntax: inline and update InstanceTemplate
- Include name and type of stream parameters in the Mir

### Added
- `get()` and `is_fresh()` stream access
- Implemented `Display` for Mir to display the whole specification
- `ty()` and `name()` methods for Stream
- Added `dependency_graph()` to generate dependency graph as json or in dot-format
- `Window` trait for shared information of sliding and discrete window

## [0.5.3] - 12.04.2022

### Changed
- Use new version of Hir crate

## [0.5.2] - 11.04.2022

### Fixed
- Small bug when using spawn / filter / close

## [0.5.1] - 11.04.2022

### Fixed
- Bug when lowering `hir::ArithLogOp::Shl`

## [0.5.0] - 2021-11-15

### Added
- support for parameterization

### Changed
- Error are now collected and returned, not directly emitted.
- Most interface functions do not depend on the Handler anymore.

## [0.4.4] - 2021-09-30

### Added
- New window operations: last, median, nth-percentile, variance, standard deviation, covariance

## [0.4.3] - 2021-08-03

### Changed
- lowering of cast from function to convert

### Fixed
- Sqrt and other functions expected value of size 0

## [0.4.2] - 2021-05-27

### Added
- A trigger can now be annotated with streams. Their values are incorporated into the trigger message.

## [0.4.1] - 2021-05-24

### Changed
- Minor changes to the MIR to be compatible with the interpreter.

## [0.4.0] - 2021-04-08

### Changed
- Complete revision of frontend
- Distribution of code into several sub-crates

## [0.3.5] - 2021-01-07

### Added 
- Updated num and uom dependency.

## [0.3.4] - 2020-09-18

### Added 
- Frontend can now process discrete sliding windows.

## [0.3.3] - 2020-07-22

### General
- Libpcap now only loaded, if the network monitoring interface is used.

## [0.3.2] - 2020-04-27

### Added
- Frontend: `hold(or: default)` syntax now supported
- Evaluator: Added `forall` (`conjunction`) and `exists` (`disjunction`) window aggregations over boolean streams
- Evaluator: Implemented `sum` window aggregation over boolean streams

## [0.3.1] - 2020-03-05

### Fixed
- Evaluator: Fixed Evaluation of Min/Max

## [0.3.0] - 2020-02-27
### Added
- Evaluator: Add pcap interface (see `ids` subcommand) (requires libpcap dependency)
- Frontend: Add pcap interface (see `ids` subcommand) (requires libpcap dependency)
- Frontend: Add Float16 Type
- Language: Add bitwise operators and `&`, or `|`, xor `^`, left shift `<<`, and right shift `>>`
- Language: Add `Bytes` data type
- Language: Add method `Bytes.at(index:)`, e.g., `bytes.at(index: 0)`

### Fixed
- Evaluator: Fixed Min/Max aggregation function implementation

## [0.2.0] - 2019-08-23
### Added
- Language: it is now possible to annotate optional types using the syntax `T?` to denote optional of type `T`, e.g., `Int64?`
- Language: support `min` and `max` as aggregations for sliding windows, e.g., `x.aggregate(over: 1s, using: min)`

### Fixed
- Frontend: Fix parsing problem related to keywords, e.g., `output outputxyz` is now be parsed (was refused before)
- Frontend: Ignore [BOM](https://de.wikipedia.org/wiki/Byte_Order_Mark) at the start of specification file
- Interpreter: Fix sliding window aggregation bug


## [0.1.0] - 2019-08-12
### Added
- Initial public release: parsing, type checking, memory analysis, lowering, and evaluation of StreamLAB specifications

