import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed
import argparse
from pyctcdecode import build_ctcdecoder
from multiprocessing import Pool
from tqdm import tqdm
import re
import json
import jiwer
import os

from datasets import DatasetDict
from datasets import Dataset
import pandas as pd
import unidecode

resampler = torchaudio.transforms.Resample(48_000, 16_000)


class KenLM:
    def __init__(self, tokenizer, model_name, num_workers=8, beam_width=128):
        self.num_workers = num_workers
        self.beam_width = beam_width
        vocab_dict = tokenizer.get_vocab()
        self.vocabulary = [
            x[0] for x in sorted(vocab_dict.items(), key=lambda x: x[1], reverse=False)
        ]
        # Workaround for wrong number of vocabularies:
        if tokenizer.name_or_path == "lucio/wav2vec2-large-xlsr-luganda":
            self.vocabulary += ["_", "-"]
            self.vocabulary[1] = ""  # Remove apostrophe
        elif tokenizer.name_or_path == "lucio/wav2vec2-large-xlsr-kinyarwanda":
            self.vocabulary += ["_"]
        else:
            self.vocabulary = self.vocabulary[:-2]
        self.decoder = build_ctcdecoder(self.vocabulary, model_name)

    @staticmethod
    def lm_postprocess(text):
        return " ".join([x if len(x) > 1 else "" for x in text.split()]).strip()

    def decode(self, logits):
        probs = logits.cpu().numpy()
        # probs = logits.numpy()
        with Pool(self.num_workers) as pool:
            text = self.decoder.decode_batch(pool, probs)
            text = [KenLM.lm_postprocess(x) for x in text]
        return text


chars_to_ignore = [
    ",",
    "?",
    ".",
    "!",
    "-",
    ";",
    ":",
    '""',
    "%",
    "'",
    '"',
    "�",
    "'",
    "\u2018",
    "\u2019",
]

chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'


def remove_special_characters(batch):
    # word-internal apostrophes are marking contractions
    batch["norm_text"] = re.sub(r"[‘’´`]", r"'", batch["sentence"])
    # most other punctuation is ignored
    batch["norm_text"] = (
        re.sub(chars_to_ignore_regex, "", batch["norm_text"]).lower().strip()
    )
    batch["norm_text"] = re.sub(r"(-|' | '|  +)", " ", batch["norm_text"])
    # remove accents from a few characters (from loanwords, not tones)
    batch["norm_text"] = unidecode.unidecode(batch["norm_text"])
    return batch


def speech_file_to_array_fn(batch):
    try:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        # batch["sampling_rate"] = 16_000
        # batch["target_text"] = batch["sentence"]

        return batch
    except Exception as e:
        print(f"Could not process file {batch['path']}. Error: {str(e)}")

        # batch["speech"] = None

        return batch


def load_datasets(language, dataset_data_dir, label_file_path):
    data = []
    lang = language
    root_dir = dataset_data_dir
    label_file_path = label_file_path

    with open(label_file_path, "r") as f:
        labels = json.load(f)[lang]

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ogg"):
                filepath = subdir + os.sep + file
                subdir_id = os.path.basename(subdir)
                if (
                    subdir_id in labels.keys()
                    and filepath.split("dataset")[1][1:] in labels[subdir_id]["audio"]
                ):
                    transcript = labels[subdir_id]["transcript"]
                    data.append(
                        {
                            "path": filepath.split("dataset")[1][1:],
                            "sentence": transcript,
                        }
                    )

    dataset = Dataset.from_pandas(pd.DataFrame(data))

    train_test_valid = dataset.train_test_split(test_size=0.1)

    test_valid = train_test_valid["test"].train_test_split(test_size=0.5)

    train_test_valid_dataset = DatasetDict(
        {
            "train": train_test_valid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    train_dataset = train_test_valid_dataset["train"]
    test_dataset = train_test_valid_dataset["test"]
    eval_dataset = train_test_valid_dataset["valid"]

    return train_dataset, test_dataset, eval_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language = "luganda"
    dataset_data_dir = "/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/dataset/data/luganda"
    label_file_path = "/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/wav2vec/salt-multilingual-audio-transcript.json"
    model_name = "/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/wav2vec/output/wav2vec2-base-lug"

    wer = load_metric("wer")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    train_set, test_set, val_set = load_datasets(
        language, dataset_data_dir, label_file_path
    )
    test_set = test_set.map(speech_file_to_array_fn)
    test_set = test_set.map(remove_special_characters)
    model = model.to(device)

    kenlm = None
    if kenlm is not None:
        kenlm = KenLM(processor.tokenizer, "5gram.bin")

    def evaluate(batch):
        inputs = processor(
            batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
            ).logits

        if kenlm:
            print("Decoding with KenLM")
            batch["pred_strings"] = kenlm.decode(logits)
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            batch["pred_strings"] = processor.batch_decode(predicted_ids)

        return batch

    result = test_set.map(evaluate, batched=True, batch_size=16)

    WER = 100 * wer.compute(
        predictions=result["pred_strings"], references=result["norm_text"]
    )
    print(f"WER: {WER:.2f}%")

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
