# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - TBD

### Added
- Added the `FeatureSelector` to check for unsupported language features.
- Added `is_eval_filtered` method to stream trait.
- Add '->' syntactic sugar

### Changed
- Applying the `abs()` function to unsigned values is now rejected by the typechecker.
- Discrete windows are now event-based and follow the pacing of the stream they aggregate.
- Projecting out of optional tuples is now allowed and produces an optional value.
- Include origin of stream access in `direct_accesses_with` and `direct_accessed_by_with`.

## [0.4.1] - 02.02.2023

### Changed
- adapt to new rust version
- update Copyright in `README.md`

## [0.4.0] - 19.12.2022

### Changed
- Include access to stream access kind in Hir
- Ignore mirror streams field in AST, because they represent syntactic sugar artifacts.
- Update for new Syntax: inline InstanceTemplate in Hir Output streams

### Added
- `get()` and `is_fresh()` stream access

## [0.3.3] - 12.04.2022

### Fixed
- Bug with semantic type equality

## [0.3.2] - 11.04.2022

### Fixed
- Small bug when using spawn / filter / close

## [0.3.1] - 11.04.2022

### Changed
- Changes for syntactic sugar base
- Semantic types now match pseudo-semantically based on Conjunctions and Disjunctions

## [0.3.0] - 2021-11-15

### Added
- support for parameterization

### Changed
- Error are now collected and returned, not directly emitted.
- Most interface functions do not depend on the Handler anymore.

## [0.2.0] - 2021-09-30

### Added
- New window operations: last, median, nth-percentile, variance, standard deviation, covariance
- Added StreamSelector to select classes of streams
- Added convenience methods to ConcretePacingType

## [0.1.4] - 2021-08-03

### Added
- trigonometric functions: tan, arcsin, arccos
- Convenience methods for parameterized streams to MIR

### Fixed
- Streams can no longer have optional value types

## [0.1.3] - 2021-06-08

### Renamed
- Renamed activation condition to pacing type

## [0.1.2] - 2021-05-27

### Added
- A trigger can now be annotated with streams. Their values are incorporated into the trigger message.

## [0.1.1] - 2021-05-24

### Changed
- Minor changes to the MIR to be compatible with the interpreter, including the parametrization of streams 

## [0.1.0] - 2021-04-08

- Initial public release
