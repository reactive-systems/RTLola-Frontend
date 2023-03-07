# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - ???

### Added
- add `offset(by: off, or: dft)` syntactic sugar
- add `->` syntactic sugar

## [0.3.1] - 02.02.2023

### Changed
- adapt to new rust version
- adapt to new pest version and replaced `PrecClimer` with `PrattParser`
- update Copyright in `README.md`

## [0.3.0] - 19.12.2022

### Changed 

- Update Parser for new Syntax: ```output o (p) spawn @i when i with 3 eval @1Hz when a > 0 with p+a.hold() close @10Hz when !i```

### Added
- `get()` and `is_fresh()` stream access

## [0.2.1] - 2022-04-11

### Added
- Added basic structure for syntactic sugar
- Syntactic sugar:
    - method shorthand for aggregations
    - last method for offset of -1
    - mirror streams
    - delta function

## [0.2.0] - 2021-11-15

### Added
- support for parameterization

### Changed
- Error are now collected and returned, not directly emitted.
- Most interface functions do not depend on the Handler anymore.

## [0.1.4] - 2021-09-30

### Added
- New window operations: last, median, nth-percentile, variance, standard deviation, covariance

## [0.1.3] - 2021-06-08

### Renamed
- Renamed activation condition to pacing type

## [0.1.2] - 2021-05-27

### Added
- A trigger can now be annotated with streams. Their values are incorporated into the trigger message.

## [0.1.1] - 2021-05-24

### Changed
- Changed repository url

## [0.1.0] - 2021-04-08

- Initial public release
