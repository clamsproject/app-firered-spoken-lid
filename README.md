# FireRed Spoken LID

## Description

Spoken language identification (LID) over fixed-length windows of an audio track, using [FireRedLID](https://huggingface.co/FireRedTeam/FireRedLID), the language identification model of [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S).

FireRedLID gives only one label for a whole audio clip, and cannot find where the language changes inside the clip. The app therefore feeds it fixed-length clips: it reads the audio of an `AudioDocument` or `VideoDocument` in back-to-back windows through `mmif.utils.audio_document_helper`, and runs FireRedLID on each window. The window length is the runtime parameter `windowMs`.

FireRedLID gives one label per window: a language, or `other`. The app writes languages as BCP 47 language tags: the two-letter ISO 639-1 code where one exists, else the three-letter ISO 639-3 code (`ceb`, `haw`, `sco`, `war`). FireRedLID's own `iw` and `jw` become `he` (Hebrew) and `jv` (Javanese). For Chinese, FireRedLID also gives a dialect, which the app drops, so the label is `zh`. Consecutive windows with the same label become one `TimeFrame`, with `start` and `end` in milliseconds. Its `scores` list holds, in time order, the FireRedLID confidence of that label in each of the merged windows, so a low-confidence window inside a long segment stays visible. With `tfMerge=false`, each window becomes its own `TimeFrame`. The view's `TimeFrame` metadata lists all labels of the model as `labelset`.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

The app runs on CPU or, when available, a CUDA GPU. The model weights (about 3.6 GB) are downloaded on first use to the Hugging Face cache (`$HF_HOME`).

The FireRedLID inference code comes from the [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) repository, a git submodule at a fixed commit. Clone this repository with `git clone --recurse-submodules`, or run `git submodule update --init` in an existing clone. The FireRedASR2S Python package itself is not installed, since it requires Python 3.11 or later and loads the whole ASR system.

### Configurable runtime parameters

For the full list of parameters, see the app metadata in the [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py).

## Development

Run the tests with `pytest` from the repository root, with the submodule checked out. The tests replace the model with a stub, so no model download is needed.
