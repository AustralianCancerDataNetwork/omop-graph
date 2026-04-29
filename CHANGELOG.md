## [1.0.4](https://github.com/AustralianCancerDataNetwork/omop-graph/compare/v1.0.3...v1.0.4) (2026-04-29)


### Bug Fixes

* Correct on-demand embedding calculation ([#7](https://github.com/AustralianCancerDataNetwork/omop-graph/issues/7)) ([1495def](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/1495defc816415974e857472604f61e4894e9536))

## [1.0.3](https://github.com/AustralianCancerDataNetwork/omop-graph/compare/v1.0.2...v1.0.3) (2026-04-22)


### Bug Fixes

* Support new omop-emb structure and interfaces ([#5](https://github.com/AustralianCancerDataNetwork/omop-graph/issues/5)) ([85c1fb7](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/85c1fb7f55dd8dd633936ddcd57d69dc277dd6a3))

## [1.0.2](https://github.com/AustralianCancerDataNetwork/omop-graph/compare/v1.0.1...v1.0.2) (2026-04-15)


### Bug Fixes

* Remove Limit from pipeline ([2f44bc4](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/2f44bc4b47db9d9243bbab1bf330cd7412562c0a))

## [1.0.1](https://github.com/AustralianCancerDataNetwork/omop-graph/compare/v1.0.0...v1.0.1) (2026-04-15)


### Bug Fixes

* Perform sorting and limits in queries ([52cf7d2](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/52cf7d22149612aeeb7e29f2a2dcee7d0fbe0b3a))

# 1.0.0 (2026-04-13)


### Bug Fixes

* **ci:** Chore/semantic release setup ([#2](https://github.com/AustralianCancerDataNetwork/omop-graph/issues/2)) ([6745d35](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/6745d3576db1adfd2b42a6f0ebfc599f1f5f546e))
* **ci:** stabilize release and publish workflows ([5a276aa](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/5a276aad11cb658b95eab8740dd38b671273477a))


### Features

* overhaul embedding-grounding flow, benchmark tooling, expanded tests, and docs refresh ([c42de2f](https://github.com/AustralianCancerDataNetwork/omop-graph/commit/c42de2f22da6ece2f3472e84e8f9f2e88fc439ed))

## 0.1.1
- additional functionality for supporting phenotype simplification by parent grouping

## 0.1.2
- added roots, leaves, singletons and altlabel queries to knowledge graph to support oaklib interface methods downstream

## 0.1.3
- bugfix for synonym labels

## 0.2.0
- update to use refactored omop-alchemy base

## 0.2.1
- predicate types need to be actual booleans why are they coming from strings in relationship table?
