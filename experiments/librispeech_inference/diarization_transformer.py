import argparse
import copy
import json
import os
from typing import Dict, List

from experiments.configure.gen_synthetic_test_tables import synthetic_config_path
from experiments.train.train import configure_model, configure_trainer, parse_train_config
from experiments.train.transformer_data_module import TransformerDataModule
from models.diarization_transformer import Recognition


def configure_single_speaker_validation_configs(librispeech_path: str, tables: List[str]) -> List:
    val_config = []
    for table in tables:
        val_config.append({
            'librispeech_path': os.path.join(librispeech_path, table),
            'speakers_num': 1,
            'name': f'{table}-1sp',
        })
    return val_config


def configure_synthetic_validation_configs(librispeech_path: str, tables: List[str],
                                           synthetic_overlaps: List[float]) -> List:
    val_config = []
    for table in tables:
        for overlap in synthetic_overlaps:
            val_config.append({
                'librispeech_path': os.path.join(librispeech_path, table),
                'sample_pairs_config': synthetic_config_path(librispeech_path, table),
                'speakers_num': 2,
                'overlap': overlap,
                'name': f'{table}-2sp-{overlap}overlap',
            })
    return val_config


def configure_validation_configs(librispeech_path: str, tables: List[str], synthetic_overlaps: List[float]) -> List:
    return configure_single_speaker_validation_configs(librispeech_path, tables) + \
           configure_synthetic_validation_configs(librispeech_path, tables, synthetic_overlaps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_config_path', type=str,
                        default='experiments/configs/train_configs/diarization_transformer_parallel.json')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to diarization transformer model checkpoint')
    parser.add_argument('--librispeech_path', type=str, default='LibriSpeech')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')

    args = parser.parse_args()

    train_config = parse_train_config(args.train_config_path)
    data_module = TransformerDataModule(train_config)

    transformer = configure_model(args.checkpoint_path, train_config,
                                  data_module.validation_table_names,
                                  data_module.test_table_names)

    trainer = configure_trainer(train_config)

    test_recognitions: Dict[str, List[Recognition]] = {}
    for i, (table_name, test_dataloader) in enumerate(zip(data_module.test_table_names,
                                                          data_module.test_dataloader())):
        trainer.test_idx = i
        trainer.test(dataloaders=test_dataloader, model=transformer)
        for key, val in trainer.test_recognitions.items():
            test_recognitions[key] = copy.deepcopy(val)

    with open(args.output_path, 'w') as f:
        json.dump({name: [r.to_json() for r in recs] for name, recs in test_recognitions.items()}, f)
