import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
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
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import yaml


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
    batch["audio"] = None  # Initialize 'speech' key with default value
    if batch["path"] is not None:
        try:
            speech_array, sampling_rate = torchaudio.load(batch["path"])
            audio = resampler(speech_array).squeeze().numpy()
            batch["audio"] = audio
            return batch
        except Exception as e:
            print(f"Could not process file {batch['path']}. Error: {str(e)}")
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


def batch_transcribe(test_set, model_name, device, chunk_length_s=30, batch_size=16):
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=chunk_length_s,
        device=device,
        batch_size=batch_size,
    )

    # Process audio clips individually
    results = []
    for item in tqdm(test_set, desc="Transcribing"):
        audio_data = np.array(item["audio"])
        transcription = asr_pipe(audio_data)
        results.append(transcription)

    return results


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    language = "acholi"
    model_type = "whisper"
    whisper_model = f"akera/whisper-small-{language}"
    wav2vec_model = "ak3ra/wav2vec2-sunbird-speech-ach"
    dataset_data_dir = f"/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/dataset/data/{language}"
    label_file_path = "/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/wav2vec/salt-multilingual-audio-transcript.json"

    train_set, test_set, val_set = load_datasets(
        language, dataset_data_dir, label_file_path
    )
    test_set = test_set.map(speech_file_to_array_fn)
    test_set = test_set.map(remove_special_characters)
    test_set = test_set.filter(
        lambda example: example["audio"] is not None and example["sentence"] is not None
    )

    wer = load_metric("wer")

    transcriptions = batch_transcribe(test_set, wav2vec_model, device)
    references = [item["norm_text"] for item in test_set]
    transcribed_texts = [item["text"] for item in transcriptions]

    WER = 100 * wer.compute(predictions=transcribed_texts, references=references)

    print(f"WER: {WER:.2f}%")


if __name__ == "__main__":
    main()
