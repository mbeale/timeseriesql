# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2020-01-18

### Added
- Added generic class to build plotting libraries against the TimeSeries class

### Removed
- removed the AppOptics backend into it's own repo

## [0.1.4] - 2020-01-11

### Added
- query object can use `range` to set specific time windows
- appoptics added modulo and power binary operations
- query class has 100% code coverage

## [0.1.3] - 2019-12-07
### Added
- improved label handling/reduction
- use a more intuitive AST for decompiling generators
- 100% test coverage for AST
- added a fill function
- added tests for PCA and differencing
- appoptics added support for != in filters
- appoptics added raw composite

### Removed
- old object based decompiler

## [0.1.2] - 2019-11-08
### Added
- added default value when merging time series
- added string time period to resample
- vectorized created time index

### Removed
- Plot function

## [0.1.1] - 2019-10-20
### Added
- Added function processing for AppOptics
- Added ability to decompile functions with/without args and/or kwargs

