# Changelog

## [1.0.0] - 2026-07-11

- Migrated Whisper to configuration assets plus isolated request-driven runtimes.
- Added caller-selected backends and main-thread audio preparation.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-05

### Added
- Initial release
- WhisperEngine for OpenAI Whisper models
- Whisper tokenizer support for Whisper vocabulary handling
- WhisperModelSet for grouping encoder/decoder models
- Transcription with and without timestamps
- Force alignment support
- Multi-language support with auto-detection
- VoiceActivityDetector for audio segmentation
