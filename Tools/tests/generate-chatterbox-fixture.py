#!/usr/bin/env python3
"""Generate deterministic Chatterbox Turbo/Nano parity tensors from the official pipeline."""

import argparse
import base64
import gzip
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

SAMPLE_RATE = 24_000
START_SPEECH = 6561
STOP_SPEECH = 6562
SILENCE = 4299


def prepare_text(text: str) -> str:
    if not text:
        return "You need to add some text for me to talk."
    text = " ".join(text.split())
    for old, new in (
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip()
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?", "-", ",")):
        text += "."
    return text


def encode_tensor(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path.as_posix(), options, providers=["CPUExecutionProvider"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Pinned source as repository@revision. Repeat for implementation, checkpoint, and export.",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--voice", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text", default="Oh, that's hilarious! [chuckle]")
    parser.add_argument("--embed", required=True)
    parser.add_argument("--speech-encoder", required=True)
    parser.add_argument("--language-model", required=True)
    parser.add_argument("--decoder", required=True)
    args = parser.parse_args()

    root = args.model_dir or Path(snapshot_download(args.repository, revision=args.revision))
    tokenizer = AutoTokenizer.from_pretrained(root)
    normalized_text = prepare_text(args.text)
    input_ids = tokenizer(normalized_text, return_tensors="np")["input_ids"].astype(np.int64)

    embed = session(root / args.embed)
    speech_encoder = session(root / args.speech_encoder)
    language_model = session(root / args.language_model)
    decoder = session(root / args.decoder)

    inputs_embeds = embed.run(None, {"input_ids": input_ids})[0]
    audio, _ = librosa.load(args.voice, sr=SAMPLE_RATE)
    audio = audio[np.newaxis, :].astype(np.float32)
    speech_names = [output.name for output in speech_encoder.get_outputs()]
    speech_values = speech_encoder.run(None, {"audio_values": audio})
    speech = dict(zip(speech_names, speech_values))

    combined = np.concatenate((speech["audio_features"], inputs_embeds), axis=1)
    batch_size, sequence_length, _ = combined.shape
    language_inputs = {
        "inputs_embeds": combined,
        "attention_mask": np.ones((batch_size, sequence_length), dtype=np.int64),
        "position_ids": np.arange(sequence_length, dtype=np.int64).reshape(1, -1),
    }
    for model_input in language_model.get_inputs():
        if not model_input.name.startswith("past_key_values."):
            continue
        shape = [batch_size, int(model_input.shape[1]), 0, int(model_input.shape[3])]
        dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
        language_inputs[model_input.name] = np.zeros(shape, dtype=dtype)

    language_names = [output.name for output in language_model.get_outputs()]
    language_values = language_model.run(None, language_inputs)
    language = dict(zip(language_names, language_values))
    logits = language["logits"][:, -1, :].copy()
    start_score = logits[:, START_SPEECH]
    logits[:, START_SPEECH] = np.where(start_score < 0, start_score * 1.2, start_score / 1.2)
    first_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)

    generated = first_token if first_token.item() != STOP_SPEECH else np.empty((1, 0), dtype=np.int64)
    decoder_tokens = np.concatenate(
        (
            speech["audio_tokens"],
            generated,
            np.full((1, 3), SILENCE, dtype=np.int64),
        ),
        axis=1,
    )
    waveform = decoder.run(
        None,
        {
            "speech_tokens": decoder_tokens,
            "speaker_embeddings": speech["speaker_embeddings"],
            "speaker_features": speech["speaker_features"],
        },
    )[0]

    sources = []
    for source in args.source:
        repository, revision = source.rsplit("@", 1)
        sources.append({"repository": repository, "revision": revision})

    fixture = {
        "repository": args.repository,
        "revision": args.revision,
        "sources": sources,
        "text": args.text,
        "normalizedText": normalized_text,
        "tensors": {
            "input_ids": encode_tensor(input_ids),
            "inputs_embeds": encode_tensor(inputs_embeds),
            "audio_features": encode_tensor(speech["audio_features"]),
            "audio_tokens": encode_tensor(speech["audio_tokens"]),
            "speaker_embeddings": encode_tensor(speech["speaker_embeddings"]),
            "speaker_features": encode_tensor(speech["speaker_features"]),
            "first_token": encode_tensor(first_token),
            "decoder_speech_tokens": encode_tensor(decoder_tokens),
            "waveform": encode_tensor(waveform),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(fixture, stream, separators=(",", ":"))


if __name__ == "__main__":
    main()
