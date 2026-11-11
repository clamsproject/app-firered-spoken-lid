import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
from mmif import AnnotationTypes, Document, DocumentTypes, Mmif

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

RATE = 16000
SECONDS = 3.5
# label of each 1000 ms window of the test audio
TOPS = ['en', 'en', 'fr', 'en']
VOCAB = ['<blank>', '<unk>', '<pad>', '<sos>', '<eos>', 'en', 'es', 'fr',
         'zh', 'other', 'mandarin', 'iw', 'jw']


class FakeLid:
    """Stands in for a loaded ``FireRedLid``: records batch sizes and the
    sample scale, and decodes the labels of ``tops`` in window order."""

    def __init__(self, tops=TOPS):
        self.tops = tops
        self.batch_sizes = []
        self.peaks = []
        self.seen = 0
        self.config = SimpleNamespace(
            use_gpu=False, beam_size=3, nbest=1, decode_max_len=2,
            softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0)
        self.tokenizer = SimpleNamespace(dict=_Dict(VOCAB))
        self.model = SimpleNamespace(process=self.process)

    def feat_extractor(self, wavs, ids):
        # 25 ms frames every 10 ms, as FireRedLID's fbank; a shorter
        # window has none and is dropped
        kept = [(rate, wav, uttid) for (rate, wav), uttid in zip(wavs, ids)
                if len(wav) >= rate * 25 // 1000]
        self.peaks.extend(float(np.abs(wav).max()) for _, wav, _ in kept)
        if not kept:
            return None, None, [], [], []
        frames = [1 + (len(wav) - 400) // 160 for _, wav, _ in kept]
        feats = torch.zeros(len(kept), max(frames), 80)
        return (feats, torch.tensor(frames), None, None,
                [uttid for _, _, uttid in kept])

    def process(self, feats, lengths, *args):
        self.batch_sizes.append(len(feats))
        nbests = []
        for _ in range(len(feats)):
            i = self.seen
            label = self.tops[i % len(self.tops)]
            nbests.append([{
                'yseq': torch.tensor([VOCAB.index(label)]),
                'confidence': torch.tensor(0.6 + 0.1 * (i % 3),
                                           dtype=torch.float64)}])
            self.seen += 1
        return nbests


class _Dict:
    def __init__(self, id2word):
        self.id2word = id2word

    def __getitem__(self, key):
        return self.id2word[key]


@pytest.fixture
def lid(monkeypatch):
    fake = FakeLid()
    monkeypatch.setattr(app._FireRedLid, 'load',
                        classmethod(lambda cls: cls(fake)))
    return fake


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(int(RATE * SECONDS)) / RATE
    samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype('int16')[None, :]
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('pcm_s16le', rate=RATE, layout='mono')
        frame = av.AudioFrame.from_ndarray(samples, format='s16',
                                           layout='mono')
        frame.sample_rate = RATE
        frame.pts = 0
        frame.time_base = Fraction(1, RATE)
        out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def run(audio_path, **params):
    mmif = Mmif(validate=False)
    mmif.add_document(Document({
        '@type': DocumentTypes.AudioDocument,
        'properties': {'id': 'd1', 'mime': 'audio/wav',
                       'location': f'file://{audio_path}'}}))
    params = {k: [str(x) for x in v] if isinstance(v, list) else [str(v)]
              for k, v in params.items()}
    return Mmif(app.get_app().annotate(mmif, **params))


def timeframes(out):
    view = out.get_view_contains(AnnotationTypes.TimeFrame)
    tfs = list(view.get_annotations(AnnotationTypes.TimeFrame))
    return view, [(tf.get_property('start'), tf.get_property('end'),
                   tf.get_property('label'), tf.get_property('scores'))
                  for tf in tfs]


def test_segments():
    windows = [(0, 10, 'en', .9), (10, 20, 'en', .8), (20, 25, 'fr', .7)]
    assert list(app._segments(windows)) == [
        (0, 20, 'en', [.9, .8]), (20, 25, 'fr', [.7])]
    assert list(app._segments(windows, merge=False)) == [
        (0, 10, 'en', [.9]), (10, 20, 'en', [.8]), (20, 25, 'fr', [.7])]
    # windows that do not touch stay apart
    assert list(app._segments([(0, 10, 'en', .9), (15, 20, 'en', .8)])) == [
        (0, 10, 'en', [.9]), (15, 20, 'en', [.8])]


def test_same_language_windows_are_merged(lid, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000))
    assert tfs == [(0, 2000, 'en', [0.6, 0.7]), (2000, 3000, 'fr', [0.8]),
                   (3000, 3500, 'en', [0.6])]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['document'] == 'd1'
    assert contain['timeUnit'] == 'milliseconds'
    # special and dialect tokens are not labels, and FireRedLID's legacy
    # codes become BCP 47 tags
    assert contain['labelset'] == ['en', 'es', 'fr', 'zh', 'other', 'he',
                                   'jv']


def test_legacy_codes_become_bcp47_tags(monkeypatch, audio_path):
    monkeypatch.setattr(app._FireRedLid, 'load',
                        classmethod(lambda cls: cls(FakeLid(['iw', 'jw']))))
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False))
    assert [label for _, _, label, _ in tfs] == ['he', 'jv', 'he', 'jv']


def test_samples_reach_the_model_at_int16_scale(lid, audio_path):
    run(audio_path, windowMs=1000)
    # the tone peaks at 10000 of 32767 in the file
    assert all(9000 < peak < 11000 for peak in lid.peaks)


def test_merge_off_gives_one_timeframe_per_window(lid, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False))
    assert [(s, e) for s, e, _, _ in tfs] == [
        (0, 1000), (1000, 2000), (2000, 3000), (3000, 3500)]
    assert all(len(scores) == 1 for _, _, _, scores in tfs)


def test_short_last_window_is_dropped(lid, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, minLastWindowMs=600))
    assert tfs[-1][:3] == (2000, 3000, 'fr')


def test_window_too_short_for_the_model_is_skipped(lid, audio_path):
    # 3500 ms in 1160 ms windows leaves a 20 ms tail, shorter than one
    # feature frame
    _, tfs = timeframes(run(audio_path, windowMs=1160, tfMerge=False))
    assert tfs[-1][1] == 3480


def test_batches_follow_batch_size(lid, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, batchSize=3))
    assert lid.batch_sizes == [3, 1]
    assert len(tfs) == 3


# probabilities over the label tokens of VOCAB (en, es, fr, zh, other, iw,
# jw) for each 1000 ms window
PROBS = [[.6, .1, .1, .1, .1, 0, 0],
         [.3, .3, .4, 0, 0, 0, 0]]


@pytest.fixture
def first_step(monkeypatch):
    tokens = [i for i, w in enumerate(VOCAB)
              if w not in app._FireRedLid._SPECIAL_TOKENS
              | app._FireRedLid._DIALECT_TOKENS]

    def fake(model, feats, lengths):
        logits = torch.full((len(feats), len(VOCAB)), -1e9)
        for row in range(len(feats)):
            p = torch.tensor(PROBS[(fake.seen + row) % len(PROBS)])
            # softmax(logit / T) gives back p
            logits[row, tokens] = torch.log(p.clamp_min(1e-12)) * 1.25
        fake.seen += len(feats)
        return logits

    fake.seen = 0
    monkeypatch.setattr(app._FireRedLid, '_first_step_logits', fake)


def test_parse_label_map():
    labelset = ['en', 'es', 'fr']
    assert app._parse_label_map(['en', 'es:spanish'], labelset) == {
        'en': 'en', 'es': 'spanish'}
    with pytest.raises(ValueError, match='"xx" is not a label'):
        app._parse_label_map(['xx:en'], labelset)


def test_label_map_pools_probabilities(lid, first_step, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                               labelMap='en'))
    # window 2: en 0.3 against 0.7 for all other labels
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.7), ('en', 0.6), ('-', 0.7)]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['labelset'] == ['en', '-']


def test_labels_mapped_together_add_up(lid, first_step, audio_path):
    mmif = run(audio_path, windowMs=1000, tfMerge=False,
                     labelMap=['en:target', 'es:target'])
    _, tfs = timeframes(mmif)
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('target', 0.7), ('target', 0.6), ('target', 0.7), ('target', 0.6)]



def test_min_label_score_sets_the_bar(lid, first_step, audio_path):
    # window 2: en 0.3, below the default 0.5 but above 0.25
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.25))
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('en', 0.3), ('en', 0.6), ('en', 0.3)]
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.7))
    assert [label for _, _, label, _ in tfs] == ['-'] * 4


def test_separate_labels_each_need_the_score(lid, first_step, audio_path):
    # window 2: en 0.3 and es 0.3, each below 0.5
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['en', 'es']))
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.4), ('en', 0.6), ('-', 0.4)]
