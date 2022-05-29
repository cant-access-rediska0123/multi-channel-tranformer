# multi-channel-tranformer

Code to reproduce experiments from
[Multi-channel transformer: a transformer-based model for multi-speaker speech recognition](https://drive.google.com/file/d/1RSZ6GhaoWCAUtA2aRc74mAQk8zY_C5Rx/view?usp=sharing)
paper.

This code provides scripts to train and test models on synthetic LibriSpeech-based datasets described in the paper.
Available models:

* Single-channel transformer (1 speaker speech recognition)
* Multi-channel transformer (1 or 2 speakers recognition)
* Sequential transformer (1 or 2 speakers recognition)
* Sepformer-based multi-speaker speech recognition (1 or 2 speakers recognition)
* Streaming graph (real-time / long audios multi-speaker speech recognition)

# Scripts

### 1. Installation

```bash
git clone https://github.com/cant-access-rediska0123/multi-channel-tranformer.git
cd multi-channel-tranformer
mkdir checkpoints/ logs/ results/
pip install -r requirements.txt

# Download LibriSpeech dataset
# Will save audios to multi-channel-tranformer/LibriSpeech/
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-other-500.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz
tar -xvkf *.tar.gz
```

### 2. Train

* Train sentencepiece dictionary

```bash
python3 experiments/configure/train_sp_model.py
mv sp.model experiments/configs/dictionaries/sp_eng.model
```

* Train speech recognition models:

```bash
python3 experiments/train/train.py --train_config_path CONFIG_PATH
```

Available train configs (```CONFIG_PATH```):

* Single-channel transformer: ```experiments/train_configs/transformer.json```
* Multi-channel transformer: ```experiments/train_configs/diarization_transformer_parallel.json```
* Sequential transformer: ```experiments/train_configs/diarization_transformer_sequential.json```

This train script will save model checkpoint in  ```checkpoints/``` directory.

Current resources configuation requires computer with 8 GPUs to train models. This configuration can be modified
in ```experiments/configs/resources/8a100.json```.

### 3. Testing

The following scripts will produce json files with testing results in ```results/``` for 4 provided models. The scripts
require checkpoint files (```.ckpt```) for 3 provided models (see 'Train').

* For small audios (~40sec):

```bash
python3 experiments/librispeech_inference/transformer.py \
  --checkpoint_path SINGLE_CHANNEL_TRANSFORMER_CKPT \
  --output_table results/single-channel-transformer-results.json
python3 experiments/librispeech_inference/diarization_transformer.py \
  --checkpoint_path MULTI_CHANNEL_TRANSFORMER_CKPT \
  --output_table results/multi-channel-transformer-results.json
python3 experiments/librispeech_inference/diarization_transformer.py \
  --checkpoint_path SEQUENTIAL_TRANSFORMER_CKPT \
  --output_table results/multi-channel-transformer-results.json
python3 experiments/librispeech_inference/sepformer_baseline.py \
  --checkpoint_path SINGLE_CHANNEL_TRANSFORMER_CKPT \
  --output_table results/sepformer-baseline-results.json
```

* Real-time / long audios multi-speaker speech recognition

```bash
python3 experiments/streaming_inference/synthetic_librispeech_streaming_graph.py \
  --transformer_checkpoint_path MULTI_CHANNEL_TRANSFORMER_CKPT \
  --output_path results/streaming-results.json
```

* Recognize custom audios

The following script will recognize multi-speaker speech in custom 16kHz 1-channel audios in .wav format. Audios must be
put in ```WAVS_DIR``` directory.

```bash
python3 experiments/streaming_inference/recognize_wav.py \
  --wav_data_path WAVS_DIR \
  --transformer_checkpoint_path MULTI_CHANNEL_TRANSFORMER_CKPT \
  --output_path results/streaming-results.json
```




