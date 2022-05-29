import argparse
import os
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import List

import sentencepiece as spm

from data.text.text_processor.text_processor import TextProcessor
from data.trans import Trans
from factory.factory import make_instance


def _make_dataset(temp_file: NamedTemporaryFile, librispeech_path: str, tables: List[str], text_processor_config: str):
    text_processor: TextProcessor = make_instance(Trans, text_processor_config)

    counter = defaultdict(int)
    for table in tables:
        table_dir = os.path.join(librispeech_path, table)
        for subdir, _, files in os.walk(table_dir):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                with open(os.path.join(subdir, file)) as f:
                    lines = f.readlines()
                for line in lines:
                    line = text_processor.process_text(line) if text_processor is not None else line
                    if not line:
                        continue
                    for word in line.split(' '):
                        counter[word] += 1
                    counter[' '] += len(line.split(' ')) - 1

    for word, count in counter.items():
        temp_file.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece")

    parser.add_argument("--librispeech_path", type=str, default='LibriSpeech')
    parser.add_argument("--train_tables", action="append", type=str,
                        default=["train-clean-100", "train-clean-360", "train-other-500"])
    parser.add_argument("--text_processor_config", type=str,
                        default="experiments/configs/text_processors/eng.json")
    parser.add_argument("--remove_whitespace_from_pieces", type=bool, default=False)
    parser.add_argument("--lm_vocab_spec_tokens", type=bool, default=False)
    # SentencePiece args
    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--unk_id", type=int, default=3)
    parser.add_argument("--bos_id", type=int, default=1)
    parser.add_argument("--eos_id", type=int, default=2)
    parser.add_argument("--model_type", type=str, choices=["unigram", "bpe", "word", "char"], default="unigram")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--hard_vocab_limit", type=bool, default=True)
    parser.add_argument("--control_symbols", type=str, default="<SIL>")
    parser.add_argument("--user_defined_symbols", type=str, default="")
    parser.add_argument("--split_by_whitespace", type=bool, default=True)
    parser.add_argument("--add_dummy_prefix", type=bool, default=False)
    parser.add_argument("--remove_extra_whitespaces", type=bool, default=False)
    parser.add_argument("--max_sentencepiece_length", type=int, default=16)
    parser.add_argument("--treat_whitespace_as_suffix", type=bool, default=False)
    parser.add_argument("--character_coverage", type=float, default=1.0)
    parser.add_argument("--num_sub_iterations", type=int, default=2)

    args = parser.parse_args()

    with NamedTemporaryFile(mode="w") as f:
        _make_dataset(f, args.librispeech_path, args.train_tables, args.text_processor_config)

        s = f"--input={f.name} --input_format=tsv --model_prefix=sp"
        s += f" --pad_id={args.pad_id}"
        s += f" --bos_id={args.bos_id}"
        s += f" --eos_id={args.eos_id}"
        s += f" --unk_id={args.unk_id}"
        s += f" --model_type={args.model_type}"
        s += f" --vocab_size={args.vocab_size}"
        s += f" --hard_vocab_limit={args.hard_vocab_limit}"
        s += f" --control_symbols={args.control_symbols}"
        s += f" --user_defined_symbols={args.user_defined_symbols}"
        s += f" --split_by_whitespace={args.split_by_whitespace}"
        s += f" --add_dummy_prefix={args.add_dummy_prefix}"
        s += f" --remove_extra_whitespaces={args.remove_extra_whitespaces}"
        s += f" --max_sentencepiece_length={args.max_sentencepiece_length}"
        s += f" --treat_whitespace_as_suffix={args.treat_whitespace_as_suffix}"
        s += f" --character_coverage={args.character_coverage}"
        s += f" --num_sub_iterations={args.num_sub_iterations}"

        spm.SentencePieceTrainer.Train(s)
