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
_FIREREDASR2S_DIR = Path(__file__).resolve().parent / 'FireRedASR2S'
if not (_FIREREDASR2S_DIR / 'fireredasr2s' / 'fireredlid').is_dir():
    raise ImportError(
        f'FireRedLID code not found in {_FIREREDASR2S_DIR}; run '
        f'`git submodule update --init` in the app repository')
sys.path.insert(0, str(_FIREREDASR2S_DIR / 'fireredasr2s'))
from fireredlid import FireRedLid, FireRedLidConfig  # noqa: E402

_SAMPLE_RATE = 16000
# with a labelMap, the label of a window whose labels are mostly unmapped
_NEGATIVE_LABEL = '-'


class _FireRedLid:
    """
    FireRedLID, run on in-memory windows. Calls the parts of
    ``FireRedLid.process`` it needs, so that a failure raises instead of
    giving an empty label.
    """

    # FireRedLID computes its features from 16-bit sample values, the scale
    # of the wav files it reads
    _INT16_SCALE = 32768
    _SPECIAL_TOKENS = {'<blank>', '<unk>', '<pad>', '<sos>', '<eos>'}
    # tokens that FireRedLID adds after `zh` for the Chinese dialect
    _DIALECT_TOKENS = {'mandarin', 'yue', 'wu', 'min', 'north', 'xinan',
                       'xiang'}
    # FireRedLID codes that are not BCP 47 language tags, and their tags
    _CANONICAL_CODES = {'iw': 'he', 'jw': 'jv'}

    def __init__(self, lid):
        """:param lid: loaded ``FireRedLid``"""
        self._lid = lid
        self._device = 'cuda' if lid.config.use_gpu else 'cpu'
        # (token id, label) of each label token of the vocabulary
        self._label_tokens = [
            (i, self._canonical(token))
            for i, token in enumerate(lid.tokenizer.dict.id2word)
            if token not in self._SPECIAL_TOKENS | self._DIALECT_TOKENS]

    @classmethod
    def load(cls):
        model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        config = FireRedLidConfig(use_gpu=torch.cuda.is_available(),
                                  use_half=False)
        return cls(FireRedLid.from_pretrained(model_dir, config))

    @property
    def labels(self):
        """All labels of the model, as BCP 47 tags or ``other``."""
        return [label for _, label in self._label_tokens]

    @torch.no_grad()
    def detect(self, windows, label_map=None, min_score=0.5):
        """
        :param windows: mono ``float32`` sample arrays at 16 kHz
        :param label_map: ``{IN: OUT}``; without it, the label is
            FireRedLID's decoded label
        :param min_score: with ``label_map``, the minimum total of the best
            ``OUT`` label for it to be kept
        :returns: ``(label, score)`` per window, or ``None`` for a window too
            short for one feature frame (25 ms)
        """
        ids = [str(i) for i in range(len(windows))]
        wavs = [(_SAMPLE_RATE, samples * self._INT16_SCALE)
                for samples in windows]
        feats, lengths, _, _, kept_ids = self._lid.feat_extractor(wavs, ids)
        results = [None] * len(windows)
        if feats is None:
            return results
        feats, lengths = feats.to(self._device), lengths.to(self._device)
        if label_map:
            labeled = self._pooled_labels(feats, lengths, label_map,
                                          min_score)
        else:
            labeled = self._decoded_labels(feats, lengths)
        for uttid, result in zip(kept_ids, labeled):
            results[int(uttid)] = result
        return results

    def _canonical(self, code):
        return self._CANONICAL_CODES.get(code, code)

    def _decoded_labels(self, feats, lengths):
        """``(label, confidence)`` per input, from FireRedLID's beam search."""
        config = self._lid.config
        nbests = self._lid.model.process(
            feats, lengths, config.beam_size, config.nbest,
            config.decode_max_len, config.softmax_smoothing,
            config.aed_length_penalty, config.eos_penalty)
        for nbest in nbests:
            best = nbest[0]
            tokens = [self._lid.tokenizer.dict[int(token_id)]
                      for token_id in best['yseq'].cpu()]
            if not tokens:
                raise RuntimeError('FireRedLID decoded no label for a window')
            yield self._canonical(tokens[0]), float(best['confidence'])

    def _first_step_logits(self, feats, lengths):
        """
        Logits of the first token FireRedLID decodes, the label token, for
        the whole vocabulary: the first step of ``batch_beam_search`` in
        ``fireredlid/models/module/transformer_decoder.py``, without the
        beam.
        """
        enc_outputs, _, enc_mask = self._lid.model.encoder(feats, lengths)
        decoder = self._lid.model.lid_decoder
        ys = torch.full((len(feats), 1), decoder.sos_id, dtype=torch.long,
                        device=feats.device)
        tgt_mask = decoder.ignored_target_position_is_0(ys, decoder.pad_id)
        src_mask = enc_mask.view(len(feats), -1, enc_outputs.size(1))
        output = decoder.dropout(decoder.tgt_word_emb(ys) * decoder.scale
                                 + decoder.positional_encoding(ys))
        for layer in decoder.layer_stack:
            output = layer.forward(output, enc_outputs, tgt_mask, src_mask,
                                   cache=None)
        output = decoder.layer_norm_out(output)
        return decoder.tgt_word_prj(output[:, -1])

    def _pooled_labels(self, feats, lengths, label_map, min_score):
        """
        ``(label, probability)`` per input. The probability of each ``OUT``
        label is the sum over the labels mapped to it. The most probable
        ``OUT`` label wins if its probability is at least ``min_score``;
        otherwise the label is ``_NEGATIVE_LABEL``, with the total of the
        labels that are not mapped.
        """
        logits = self._first_step_logits(feats, lengths)
        # FireRedLID's own softmax temperature, over the label tokens only
        probs = torch.softmax(
            logits[:, [i for i, _ in self._label_tokens]]
            / self._lid.config.softmax_smoothing, dim=-1).cpu()
        for row in probs:
            pooled = {}
            for (_, label), p in zip(self._label_tokens, row.tolist()):
                target = label_map.get(label, _NEGATIVE_LABEL)
                pooled[target] = pooled.get(target, 0.0) + p
            unmapped = pooled.pop(_NEGATIVE_LABEL, 0.0)
            # empty when every label is mapped to the negative label
            best = max(pooled, key=pooled.get) if pooled else None
            if best is not None and pooled[best] >= min_score:
                yield best, pooled[best]
            else:
                yield _NEGATIVE_LABEL, unmapped


def _batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def _segments(labeled_windows, merge=True):
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


def _parse_label_map(entries, labelset):
    """
    :param entries: ``IN:OUT`` strings; an entry without a colon keeps the
        label as it is
    :returns: ``{IN: OUT}``
    :raises ValueError: if an ``IN`` label is not a label of the model
    """
    label_map = {}
    for entry in entries:
        source, _, target = (part.strip() for part in entry.partition(':'))
        if source not in labelset:
            raise ValueError(f'labelMap: "{source}" is not a label of the '
                             f'model; see the labelset in the app metadata')
        label_map[source] = target or source
    return label_map


def _mapped_labelset(label_map):
    labels = list(dict.fromkeys(label_map.values()))
    return labels + [_NEGATIVE_LABEL] * (_NEGATIVE_LABEL not in labels)


class FireredSpokenLid(ClamsApp):

    def __init__(self):
        super().__init__()
        self._model = None

    def _appmetadata(self):
        pass

    def _get_model(self):
        if self._model is None:
            self.logger.debug(f'Loading {MODEL_ID}@{MODEL_REVISION}')
            self._model = _FireRedLid.load()
        return self._model

    def _label_windows(self, windows, batch_size, label_map, min_score):
        """
        :param windows: ``(samples, start_ms, end_ms)`` per window
        :returns: ``(start_ms, end_ms, label, score)`` per window; a window
            too short for the model is left out
        """
        model = self._get_model()
        for batch in _batched(windows, batch_size):
            detected = model.detect([samples for samples, _, _ in batch],
                                    label_map, min_score)
            for (_, start_ms, end_ms), result in zip(batch, detected):
                if result is None:
                    self.logger.debug(f'Window {start_ms}-{end_ms} ms is too '
                                      f'short for FireRedLID, skipped')
                    continue
                label, score = result
                yield start_ms, end_ms, label, score

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        labels = self._get_model().labels
        label_map = _parse_label_map(parameters['labelMap'], labels)
        labelset = _mapped_labelset(label_map) if label_map else labels
        docs = chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                     mmif.get_documents_by_type(DocumentTypes.VideoDocument))
        for doc in docs:
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            windows = adh.iter_audio_windows(
                doc, parameters['windowMs'], sample_rate=_SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            labeled = self._label_windows(windows, parameters['batchSize'],
                                          label_map,
                                          parameters['minLabelScore'])
            for start_ms, end_ms, label, scores in _segments(
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
