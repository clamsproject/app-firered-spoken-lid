"""
FireRed Spoken LID: language identification over fixed-length windows of an
audio track with FireRedLID, with the windows read through the SDK's audio
document helper. Consecutive windows with the same language can be merged
into one TimeFrame.
"""
import argparse
import logging
import sys
from itertools import chain, islice
from pathlib import Path

import torch
from clams import ClamsApp, Restifier
from huggingface_hub import snapshot_download
from metadata import MODEL_ID, MODEL_REVISION
from mmif import AnnotationTypes, DocumentTypes, Mmif
from mmif.utils import audio_document_helper as adh

# FireRedLID comes from the FireRedASR2S submodule. Only its `fireredasr2s`
# directory goes on the path: importing the `fireredasr2s` package would load
# the rest of FireRedASR2S and its dependencies.
FIREREDASR2S_DIR = Path(__file__).resolve().parent / 'FireRedASR2S'
if not (FIREREDASR2S_DIR / 'fireredasr2s' / 'fireredlid').is_dir():
    raise ImportError(
        f'FireRedLID code not found in {FIREREDASR2S_DIR}; run '
        f'`git submodule update --init` in the app repository')
sys.path.insert(0, str(FIREREDASR2S_DIR / 'fireredasr2s'))
from fireredlid import FireRedLid, FireRedLidConfig  # noqa: E402

SAMPLE_RATE = 16000
# FireRedLID computes its features from 16-bit sample values, the scale of
# the wav files it reads
INT16_SCALE = 32768
SPECIAL_TOKENS = {'<blank>', '<unk>', '<pad>', '<sos>', '<eos>'}
# tokens that FireRedLID adds after `zh` for the Chinese dialect
DIALECT_TOKENS = {'mandarin', 'yue', 'wu', 'min', 'north', 'xinan', 'xiang'}
# FireRedLID codes that are not BCP 47 language tags, and their tags
CANONICAL_CODES = {'iw': 'he', 'jw': 'jv'}


def batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def segments(labeled_windows, merge=True):
    """
    :param labeled_windows: ``(start_ms, end_ms, label, score)`` per window,
        in time order
    :param merge: join consecutive windows with the same label, when each
        starts where the previous one ends
    :returns: ``(start_ms, end_ms, label, scores)`` per segment, with one
        score per window in time order
    """
    segment = None
    for start, end, label, score in labeled_windows:
        if (merge and segment is not None and segment[2] == label
                and segment[1] == start):
            segment[1] = end
            segment[3].append(score)
            continue
        if segment is not None:
            yield tuple(segment)
        segment = [start, end, label, [score]]
    if segment is not None:
        yield tuple(segment)


def load_lid():
    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    config = FireRedLidConfig(use_gpu=torch.cuda.is_available(),
                              use_half=False)
    return FireRedLid.from_pretrained(model_dir, config)


def canonical(code):
    return CANONICAL_CODES.get(code, code)


def labelset_of(lid):
    return [canonical(token) for token in lid.tokenizer.dict.id2word
            if token not in SPECIAL_TOKENS | DIALECT_TOKENS]


class FireredSpokenLid(ClamsApp):

    def __init__(self):
        super().__init__()
        self._lid = None

    def _appmetadata(self):
        pass

    def _get_lid(self):
        if self._lid is None:
            self.logger.debug(f'Loading {MODEL_ID}@{MODEL_REVISION}')
            self._lid = load_lid()
        return self._lid

    @staticmethod
    @torch.no_grad()
    def detect(lid, windows):
        """
        Runs the parts of ``FireRedLid.process`` that it needs, so that
        windows go in as arrays and a failure raises.

        :param lid: loaded ``FireRedLid``
        :param windows: mono ``float32`` sample arrays at 16 kHz
        :returns: ``(label, confidence)`` per window, or ``None`` for a
            window too short for one feature frame (25 ms)
        """
        ids = [str(i) for i in range(len(windows))]
        wavs = [(SAMPLE_RATE, samples * INT16_SCALE) for samples in windows]
        feats, lengths, _, _, kept_ids = lid.feat_extractor(wavs, ids)
        results = [None] * len(windows)
        if feats is None:
            return results
        device = 'cuda' if lid.config.use_gpu else 'cpu'
        config = lid.config
        nbests = lid.model.process(
            feats.to(device), lengths.to(device), config.beam_size,
            config.nbest, config.decode_max_len, config.softmax_smoothing,
            config.aed_length_penalty, config.eos_penalty)
        for uttid, nbest in zip(kept_ids, nbests):
            best = nbest[0]
            tokens = [lid.tokenizer.dict[int(token_id)]
                      for token_id in best['yseq'].cpu()]
            if not tokens:
                raise RuntimeError(f'FireRedLID decoded no label for window '
                                   f'{uttid} of the batch')
            results[int(uttid)] = (canonical(tokens[0]),
                                   float(best['confidence']))
        return results

    def label_windows(self, lid, windows, batch_size):
        """
        :param windows: ``(samples, start_ms, end_ms)`` per window
        :returns: ``(start_ms, end_ms, label, score)`` per window, where
            ``label`` is the decoded language and ``score`` its FireRedLID
            confidence; a window too short for the model is left out
        """
        for batch in batched(windows, batch_size):
            detected = self.detect(lid, [samples for samples, _, _ in batch])
            for (_, start_ms, end_ms), result in zip(batch, detected):
                if result is None:
                    self.logger.debug(f'Window {start_ms}-{end_ms} ms is too '
                                      f'short for FireRedLID, skipped')
                    continue
                label, score = result
                yield start_ms, end_ms, label, score

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        lid = self._get_lid()
        labelset = labelset_of(lid)
        docs = chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                     mmif.get_documents_by_type(DocumentTypes.VideoDocument))
        for doc in docs:
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            windows = adh.iter_audio_windows(
                doc, parameters['windowMs'], sample_rate=SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            labeled = self.label_windows(lid, windows,
                                         parameters['batchSize'])
            for start_ms, end_ms, label, scores in segments(
                    labeled, parameters['tfMerge']):
                view.new_annotation(AnnotationTypes.TimeFrame,
                                    start=start_ms, end=end_ms,
                                    label=label, scores=scores)
        return mmif


def get_app():
    return FireredSpokenLid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
