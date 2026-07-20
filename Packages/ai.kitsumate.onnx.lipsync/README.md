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
5. Assign the engine and configure blend shape mapping

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
Plays viseme timeline on blend shapes.

### BlendShapeMapper
Maps visemes to specific blend shape indices.

## License

MIT
