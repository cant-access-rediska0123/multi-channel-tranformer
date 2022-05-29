import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.text.dictionary.dictionary import Dictionary
from experiments.train.transformer_data_module import TransformerDataModule
from factory.factory import make_instance
from models.diarization_transformer import DiarizationTransformer

torch.multiprocessing.set_sharing_strategy('file_system')


def _load_json(json_path) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_train_config(train_config_path: str):
    train_config = _load_json(train_config_path)
    for key, value in list(train_config.items()):
        if key.endswith('_config_path') and value.endswith('.json'):
            train_config[key[:-len('_path')]] = _load_json(value)
    return train_config


def configure_model(checkpoint_path: str, train_config: Dict,
                    val_dataset_names: Optional[List[str]] = None,
                    test_dataset_names: Optional[List[str]] = None) -> DiarizationTransformer:
    transformer = DiarizationTransformer(
        transformer_config=train_config['model_config'],
        input_dim=train_config['features_config']['num-mel-bins'],
        dictionary=make_instance(Dictionary, train_config['dictionary_config']),
        val_dataset_names=val_dataset_names,
        test_dataset_names=test_dataset_names,
        optimizer_config=train_config['optimizer_config'])
    if checkpoint_path is not None:
        print(f'Loading from checkpoint: {checkpoint_path}')
        state_dict = torch.load(checkpoint_path)['state_dict']
        transformer.load_state_dict(state_dict, strict=True)
    return transformer


def configure_trainer(train_config):
    return pl.Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(**train_config['model_checkpoint_config']),
        ],
        logger=TensorBoardLogger(train_config['logs_dir']),
        gpus=train_config['resources_config']['num_gpus'],
        reload_dataloaders_every_n_epochs=1,
        **train_config['trainer_config']
    )


def _configure_all(train_config, checkpoint_path):
    datamodule = TransformerDataModule(train_config)
    transformer = configure_model(checkpoint_path, train_config,
                                  datamodule.validation_table_names,
                                  datamodule.test_table_names)
    return transformer, datamodule


def train(train_config: Dict, checkpoint_path: str):
    logs_dir = train_config['logs_dir']
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    print(f'Configured logging in {logs_dir}')
    logging.basicConfig(filename=os.path.join(logs_dir, 'train_log.txt'), level=logging.INFO)
    pl.seed_everything(42, workers=True)

    transformer, datamodule = _configure_all(train_config, checkpoint_path)

    trainer = configure_trainer(train_config)
    trainer.fit(transformer, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_path', type=str, required=True, help='Path to train config')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint to train from')

    parsed_args = parser.parse_args()

    train(parse_train_config(parsed_args.train_config_path), parsed_args.checkpoint_path)
