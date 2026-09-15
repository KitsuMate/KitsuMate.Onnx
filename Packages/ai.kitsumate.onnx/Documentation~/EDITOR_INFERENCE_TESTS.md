# Engine Inspector inference tests

Select an engine asset, assign its backend and downloaded model set, and use **Test Inference** in Edit Mode. No scene component or Play Mode is required. The first **Run** loads the engine; later runs reuse it until its configuration changes. **Unload** releases its model memory. **Cancel** is cooperative: a native inference call may need to finish before cancellation completes.

Test inputs are remembered in local Editor preferences per project and engine asset. They are not saved into the engine. Closing the Inspector, entering Play Mode, reloading scripts, or quitting releases preview resources. Multi-object testing is disabled.

## Inputs and results

- TTS: text and the selected engine's voice controls. OmniVoice offers automatic, designed, and cloned voices. Cloning requires audio and its transcript. Generation settings remain on the engine asset. Play or stop the result in the Editor, or save a PCM16 WAV.
- ASR: audio, operation, language, and alignment/VAD controls. Copy or save the transcript and available timestamps.
- Embeddings: one text per line, purpose, pooling, normalization, and token limit. Inspect vectors and pairwise cosine similarity; export all vectors as JSON.
- Lip sync: audio input and a timestamped viseme list with weights; export JSON.
- Motion: select an existing intent, duration, Humanoid rig prefab, and an `.anim` path under Assets. The intent's valid baked embedding is reused; otherwise its compatible embedding engine generates one temporarily. The intent is not modified. A temporary rig converts a single unconstrained motion segment into a Humanoid clip. There is no viewer, and existing output assets are never overwritten.

The same panels apply to installed Sentis engine variants. Backend availability and model validation are checked by the normal runtime API. Editor audio playback uses Unity's internal preview API; WAV export remains available if that API changes.

## Adding an engine family

Derive an Editor class from `InferenceEngineEditor<TRequest, TResult>` in the family's Editor assembly. Supply a serializable `TestInputs` object, `DrawTestInputs`, `CaptureRequest`, and `DrawTestResult`. `CaptureRequest` validates and snapshots inputs synchronously, then returns an async request factory for optional preparation such as embeddings. Use `AcceptResult` for output conversion and `ClearResult` to release temporary Unity objects.

Override `DrawEngineInspector` to preserve model-specific settings and download buttons. Concrete engine Editors inherit the family implementation and use small overrides for additional capabilities. Use `InferenceTestPreferences.AssetField<T>` for persistent asset references. The core Editor assembly must not reference family assemblies.

Focused model-free lifecycle tests live in `InferencePreviewTests`. Local inference uses the project’s configured model-set installations.
