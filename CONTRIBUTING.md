# Contributing to KitsuMate.Onnx

## Branches and pull requests

Create branches using this format:

```text
<type>/<scope>-<short-description>
```

For example:

```text
feat/core-backend-contracts
fix/unity-inference-int32-inputs
ci/repo-unity-smoke-workflow
```

Use lower-case kebab case. Allowed types and scopes are listed below. Open a pull
request for every change, keep its title in Conventional Commit form, and use
**squash merge**. The PR title becomes the single, release-visible commit on
`main`.

## Commit and PR-title format

```text
<type>(<scope>)!: <imperative summary>
```

The `!` is required for breaking changes. Summaries start with a verb, contain no
trailing period, and describe the observable change.

Allowed types:

- `feat` - new user-visible capability; minor release.
- `fix` - bug fix; patch release.
- `docs`, `test`, `ci`, `build`, `chore`, `refactor`, `perf` - maintenance work.

Allowed scopes:

- `core`, `onnxruntime`, `unity-inference`, `asr`, `embeddings`, `lipsync`,
  `motion`, `tts`, `tokenizers`, `ci`, `release`, `docs`, `repo`.

Examples:

```text
feat(core): add backend session contract
fix(unity-inference): reject unsupported tensor element types
ci(repo): run integration suites through GameCI
refactor(motion)!: replace legacy model compatibility metadata
```

## Release and changelog policy

All `ai.kitsumate.onnx*` packages are released as one coordinated repository
version. Do not edit `CHANGELOG.md` in feature pull requests. Release Please
derives the changelog and the release-version update from Conventional Commit
titles after they reach `main`; the repository release workflow then builds the
UPM `.tgz` assets for the resulting tag.

## Validation

Before requesting review, state what you ran in the PR description. Run relevant
package validation and Unity suites when their required fixtures are available.
Do not weaken a test merely because its external fixture is unavailable; retain
the explicit failure and document the missing input.
