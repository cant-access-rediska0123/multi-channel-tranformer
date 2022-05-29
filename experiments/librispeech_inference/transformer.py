import argparse
import copy
import json
import os
from typing import Dict, List

from tqdm import tqdm

from data.text.text import MultiSpeakerTexts
from experiments.configure.gen_synthetic_test_tables import synthetic_config_path
from experiments.train.train import configure_model, configure_trainer, parse_train_config
from experiments.train.transformer_data_module import TransformerDataModule
from models.diarization_transformer import Recognition
from torch_modules.metrics.speaker_independent_wer import SpeakerIndependentWer


def _configure_validation_configs(librispeech_path: str, tables: List[str]):
    val_config = []
    for table in tables:
        val_config.append({
            'librispeech_path': os.path.join(librispeech_path, table),
            'speakers_num': 1,
            'name': f'{table}-1sp',
        })
    return val_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument('--train_config_path', type=str,
                        default='experiments/configs/train_configs/transformer.json')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Transformer model checkpoint')
    # data config
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--synthetic_overlaps', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--output_table', type=str, required=True, help='Path to save results')

    args = parser.parse_args()

    train_config = parse_train_config(args.train_config_path)
    data_module = TransformerDataModule(train_config)
    transformer = configure_model(args.checkpoint_path, train_config,
                                  data_module.validation_table_names,
                                  data_module.test_table_names)

    trainer = configure_trainer(train_config)
    test_recognitions: Dict[str, List[Recognition]] = {}
    for i, (name, dataloader) in enumerate(zip(data_module.test_table_names, data_module.test_dataloader())):
        trainer.test_idx = i
        trainer.test(dataloaders=dataloader, model=transformer)
        for key, val in trainer.test_recognitions.items():
            test_recognitions[key] = val

    for table in copy.deepcopy(list(test_recognitions.keys())):
        print('Evaluating on synthetic', table)
        synthetic_pairs_config = synthetic_config_path(args.librispeech_path, table.split('_')[0])
        with open(synthetic_pairs_config, 'r') as f:
            synthetic_pairs = json.load(f)

        known_channel_recognitions: Dict[str, Recognition] = {r.sample_uuid: r for r in test_recognitions[table]}

        wer = SpeakerIndependentWer()
        synthetic_recognitions: List[Recognition] = []
        for sample1, sample2 in tqdm(synthetic_pairs):
            hypotheses = MultiSpeakerTexts([known_channel_recognitions[sample1].hypothesis[0],
                                            known_channel_recognitions[sample2].hypothesis[0]])
            references = MultiSpeakerTexts([known_channel_recognitions[sample1].reference[0],
                                            known_channel_recognitions[sample2].reference[0]])
            synthetic_recognitions.append(Recognition(f'{sample1}_{sample2}', hypotheses, references))
            wer.update([hypotheses], [references])
        print('cpWER:', wer.compute().item())
        for overlap in args.synthetic_overlaps:
            test_recognitions[f'synthetic_{table}_{overlap}overlap'] = synthetic_recognitions

    with open(args.output_table, 'w') as f:
        json.dump({name: [r.to_json() for r in recs] for name, recs in test_recognitions.items()}, f)
