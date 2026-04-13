## 0.3.0
- refactored grounding flow to score all candidate mappings before deduplicating to the best entry per standard concept id
- removed pre-score standard concept collapsing to preserve stronger evidence paths and label matches
- updated embedding similarity API from `unique_standard_concepts` to `standard_concepts`, including callsite and docs alignment
- improved benchmark case handling with optional per-case `parent_ids`, tuple/list/int parsing, and CLI fallback behavior
- simplified benchmark constraints (optional domain, centralized default vocabularies, cleaner vocabularies parsing)
- reworked poster benchmark output to case-first grounded reporting for easier interpretation
- expanded tests for grounding, resolver behavior, optional embedding integration, and optional full-text paths
- expanded docs and API references for grounding/resolvers/CLI and benchmark usage
- fixed docs workflow dependency installation to include `mkdocs-macros-plugin` and pinned mkdocs `<2.0` in CI

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