# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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
