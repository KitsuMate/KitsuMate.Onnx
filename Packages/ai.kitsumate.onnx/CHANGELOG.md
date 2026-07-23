# Changelog

## [Unreleased]

- Discover mixed-precision Chatterbox repositories as one model set, named after their largest graph.

## [1.0.0] - 2026-07-11

- Split immutable engine assets from isolated disposable inference runtimes.
- Added caller-owned backends, cancellation, serialized inference, stable model identity, structured validation, external model storage, and shared verified model catalogs.
- Removed mutable engine-asset lifecycle and persistent model download state.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-05

### Added
- Initial release
- Core ONNX backend abstraction and session management
- OnnxModelAsset ScriptableObject for model references
- OnnxBackend base class for backend implementations
- OnnxRuntimeBackend for ONNX Runtime integration
- InferenceEngine base class for model inference
- ModelSet base class for grouping related models
- OnnxSettings for project-wide configuration
- Addressables integration for model loading
