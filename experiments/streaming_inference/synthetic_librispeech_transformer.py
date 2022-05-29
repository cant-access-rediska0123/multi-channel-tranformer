import argparse
import json
import logging

from data.text.text_processor import MultiSpeakerTextProcessor
from data.trans import Trans
from experiments.streaming_inference.recognize_wav import parse_train_config
from experiments.streaming_inference.synthetic_librispeech_streaming_graph import SyntheticStreamingLibriSpeechSource
from factory.factory import make_instance
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--transformer_train_config_path', type=str,
                        default='experiments/configs/train_configs/diarization_transformer_parallel.json')
    parser.add_argument('--test_config_path', type=str,
                        default='experiments/configs/data/test_streaming_librispeech_1sp.json')
    parser.add_argument('--transformer_output_table', type=str, required=True, help='Path to save results')

    # evaluation
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--log_path', type=str, required='logs/log.txt')

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, level=logging.DEBUG)

    with open(args.test_config_path, 'r') as f:
        test_config = json.load(f)

    transformer_config = parse_train_config(args.transformer_train_config_path)

    text_processor: MultiSpeakerTextProcessor = make_instance(Trans, transformer_config['text_processor_config'])

    results = {}
    for table in test_config:
        print('Testing on {}...'.format(table['name']))
        with open(table['synthetic_config_path'], 'r') as f:
            synthetic_config = json.load(f)
        source = SyntheticStreamingLibriSpeechSource(args.librispeech_path, table['table_path'], synthetic_config,
                                                     transformer_output_table=args.transformer_output_table)
        hypotheses = source.transformer_hypotheses()
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
