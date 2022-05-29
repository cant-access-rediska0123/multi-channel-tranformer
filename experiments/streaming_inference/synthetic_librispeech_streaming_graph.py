import argparse
import json
import logging
import os
from typing import Dict, List, Optional

from data.asr.datasource.libri_speech.source import parse_sample
from data.asr.datasource.source import Source
from data.asr.transforms.multispeaker.utterance_mixer import UtteranceMixer
from data.audio.wave import Wave
from data.text.text import Text
from data.text.text_processor import MultiSpeakerTextProcessor
from data.trans import Trans
from data.utterance.utterance import MultiSpeakerTexts, RawMultiSpeakerUtterance
from experiments.streaming_inference.recognize_wav import StreamingDiarizationTransformerInference, parse_train_config
from factory.factory import make_instance
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer


class SyntheticStreamingLibriSpeechSource(Source[RawMultiSpeakerUtterance]):
    def __init__(self, librispeech_path: str, table_path: str, synthetic_config: Dict,
                 transformer_output_table: Optional[str] = None):
        self._table_path = table_path
        self._synthetic_config = synthetic_config
        self._librispeech_path = librispeech_path
        self._transformer_output_table = transformer_output_table

    def __len__(self):
        return len(self._synthetic_config)

    def _parse_sample(self, idx: int, s: Dict) -> RawMultiSpeakerUtterance:
        samples_to_mix: List[str] = s['samples_to_mix']
        references: List[Text] = s['reference']
        overlaps: List[float] = s['overlaps']
        mixer = UtteranceMixer(speakers_num=len(references), constant_overlap=overlaps)
        waves: List[Wave] = [parse_sample(
            os.path.join(self._librispeech_path, self._table_path, sample.split('-')[0], sample.split('-')[1]),
            sample).audio for sample in samples_to_mix]
        mixed_wave: Wave = mixer.mix_audios(waves)
        return RawMultiSpeakerUtterance(MultiSpeakerTexts(references), mixed_wave, str(idx))

    def references(self) -> Dict[str, MultiSpeakerTexts]:
        return {str(i): [t.lower() for t in s['reference']] for i, s in enumerate(self._synthetic_config)}

    def transformer_hypotheses(self) -> Dict[str, MultiSpeakerTexts]:
        assert self._transformer_output_table is not None
        with open(self._transformer_output_table, 'r') as f:
            output = json.load(f)
            recs = output[self._table_path + '_greedy']
            recognitions: Dict[str, MultiSpeakerTexts] = {r['id']: MultiSpeakerTexts(r['hypothesis']) for r in recs}
        hypotheses: Dict[str, MultiSpeakerTexts] = {}
        for idx, s in enumerate(self._synthetic_config):
            samples_to_mix = s['samples_to_mix']
            hyp: List[List[str]] = [[], []]
            first_speaker = samples_to_mix[0].split('-')[0]
            for sample in samples_to_mix:
                speaker = 0 if sample.split('-')[0] == first_speaker else 1
                hyp[speaker] += recognitions[sample.split('.')[0]]
            hypotheses[str(idx)] = MultiSpeakerTexts([Text(' '.join(hyp[speaker])) for speaker in range(2)])
        return hypotheses

    def __iter__(self):
        return iter(self._parse_sample(i, s) for i, s in enumerate(self._synthetic_config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpoints
    parser.add_argument('--transformer_train_config_path', type=str,
                        default='experiments/configs/train_configs/diarization_transformer_parallel.json')
    parser.add_argument(
        '--transformer_checkpoint_path', type=str, required=True, help='Transformer model checkpoint to test')
    parser.add_argument('--beam_size', type=int, default=1)

    parser.add_argument('--jasper_dictionary_path', type=str,
                        default='experiments/configs/dictionaries/ctc_eng.json')

    # data
    parser.add_argument('--test_config_path', type=str,
                        default='experiments/configs/data/test_streaming_librispeech_1sp.json')

    # evaluation
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--streaming_parameters_config_path', type=str,
                        default='experiments/configs/streaming/streaming_parameters.json')
    parser.add_argument('--joint_speakers_alignment', action='store_true')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--pool_size', type=int, default=24)
    parser.add_argument('--max_processed_samples', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--log_path', type=str, default='logs/log.txt')

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, level=logging.DEBUG)

    with open(args.test_config_path, 'r') as f:
        test_config = json.load(f)

    transformer_config = parse_train_config(args.transformer_train_config_path)
    with open(args.streaming_parameters_config_path, 'r') as f:
        streaming_params_config = json.load(f)

    print('Ue joint recognition:', args.joint_speakers_alignment)

    streaming_inference = StreamingDiarizationTransformerInference(
        args.transformer_checkpoint_path, transformer_config, args.beam_size,
        args.jasper_dictionary_path,
        streaming_params_config, args.joint_speakers_alignment,
        args.batch_size, args.pool_size, args.max_processed_samples, args.device)

    text_processor: MultiSpeakerTextProcessor = make_instance(Trans, transformer_config['text_processor_config'])

    results = {}
    for table in test_config:
        print('Testing on {}...'.format(table['name']))
        with open(table['synthetic_config_path'], 'r') as f:
            synthetic_config = json.load(f)
        source = SyntheticStreamingLibriSpeechSource(args.librispeech_path, table['table_path'], synthetic_config)
        hypotheses = streaming_inference(source)
        references = source.references()
        recognitions = {}
        wer = SpeakerIndependentWer()
        for key in hypotheses.keys():
            hyp, ref = hypotheses[key], text_processor.process_text(references[key])
            recognitions[key] = {'reference': ref, 'hypothesis': hyp}

            wer.update([hyp], [ref])
        cpwer = wer.compute().item()
        print('cpWER:', cpwer)
        results[table['name']] = {
            'cpWER': cpwer,
            'recognitions': recognitions,
        }

    with open(args.output_path, 'w') as f:
        json.dump(results, f)
