# KitsuMate ONNX LipSync

Lip sync and viseme detection package using ONNX models.

## Features

- **Uni2005Engine**: Phoneme detection using Allosaurus acoustic model
- **Phone-to-Viseme Mapping**: Maps detected phonemes to standard visemes
- **VisemeTimeline**: Time-stamped viseme sequence for animation

## Installation

Add to your `manifest.json`:

```json
{
  "dependencies": {
    "ai.kitsumate.onnx.lipsync": "file:../Packages/ai.kitsumate.onnx/lipsync"
  }
}
```

## Quick Start

1. Import Uni2005 ONNX acoustic model
2. Create a `Uni2005ModelSet` asset and assign model + vocabulary
3. Create a `Uni2005Engine` asset
4. Add `LipSyncPlayer` component to your character
5. Assign the engine, backend, and AudioSource
6. Add `BlendShapeVisemeTarget`, assign its root and blend-shape profile, and add it
   to the player's Targets list. Other presentations can implement `VisemeTarget`.

## Visemes

Standard 15 visemes supported:

| Viseme | Phonemes | Description |
|--------|----------|-------------|
| sil | silence | Neutral/closed |
| PP | p, b, m | Bilabial |
| FF | f, v | Labiodental |
| TH | θ, ð | Dental |
| DD | t, d, n | Alveolar |
| kk | k, g, ŋ | Velar |
| CH | tʃ, dʒ, ʃ, ʒ | Postalveolar |
| SS | s, z | Alveolar fricative |
| nn | n | Nasal |
| RR | ɹ, r | Rhotic |
| aa | ɑ, æ | Open vowel |
| E | ɛ, e | Mid-front vowel |
| ih | ɪ, i | High-front vowel |
| oh | ɔ, o | Mid-back vowel |
| ou | u, ʊ | High-back vowel |

## Components

### LipSyncPlayer
Analyzes audio and sends timed visemes to independent visual targets. Timeline
playback follows AudioSource sample position. Output includes changing weights,
silence gaps, and explicit reset on stop/disable. `OnVisemeChanged` delivers sampled
frames, even when consecutive frames use the same viseme.

Use `AnalyzeClipAsync` when another component owns audio playback, or `PlayAsync`
for standalone previews. `SetTimeline(clip, timeline)` accepts precomputed data.
The clip must match the timeline. Cancellation invalidates pending work so stale
analysis cannot start a replaced or stopped playback request.

### BlendShapeVisemeTarget
Maps visemes through a `VisemeBlendShapeProfile` and smooths transitions. Multiple
visemes mapped to one shape combine before writing the renderer. Root discovery,
profile selection, maximum weight, and smoothing live on this target rather than
on the player. `Refresh()` rebuilds bindings after changing meshes or profiles.

### VisemeTarget
Implement `ApplyViseme(VisemeFrame)` and `ResetVisemes()` for another presentation,
such as mouth sprites. Inference and audio control remain independent of rendering.

## License

MIT
