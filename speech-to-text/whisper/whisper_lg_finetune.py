from datasets import (
    Dataset,
    IterableDatasetDict,
    load_dataset,
    interleave_datasets,
    Audio,
)
import evaluate

import torch
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
import wandb
from IPython.display import clear_output
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
from huggingface_hub import notebook_login
from transformers import TrainerCallback, EarlyStoppingCallback
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
from pathlib import Path
import numpy as np
import holoviews as hv
import panel as pn
import tempfile
from bokeh.resources import INLINE

hv.extension("bokeh", logo=False)

from io import StringIO
import pandas as pd
import warnings
import jiwer
import json
import os
import torchaudio

warnings.filterwarnings("ignore")

clear_output()
torch.cuda.is_available()


from datasets import load_dataset, DatasetDict

resampler = torchaudio.transforms.Resample(48_000, 16_000)


def load_data_splits(is_streaming=True, stopping_strategy="all_exhausted"):
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "lg",
        split="train+validation",
        use_auth_token=True,
    )
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "lg", split="test", use_auth_token=True
    )

    return common_voice


class Preprocess:
    def __init__(self):
        self.augment_waveform = Compose(
            [
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.2),
                TimeStretch(
                    min_rate=0.8, max_rate=1.25, p=0.2, leave_length_unchanged=False
                ),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            ]
        )

    def speech_file_to_array_fn(self, batch):
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

    def augment_dataset(self, batch):
        audio = batch["audio"]
        if audio is not None:
            audio = np.array(audio)  # Convert list to NumPy array
            augmented_audio = self.augment_waveform(samples=audio, sample_rate=48000)
            batch["audio"] = augmented_audio
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

    return train_test_valid_dataset


dataset_name = "sunbird"


if dataset_name == "sunbird":
    language = "acholi"
    dataset_data_dir = f"/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/dataset/data/{language}"
    label_file_path = "/home/mila/a/akeraben/scratch/akera/code/sunbird/sunbird-speech/speech-to-text/wav2vec/salt-multilingual-audio-transcript.json"

    dataset_dict = load_datasets(language, dataset_data_dir, label_file_path)

    preprocessor = Preprocess()

    splits = ["train", "test", "valid"]

    for split in splits:
        dataset_dict[split] = dataset_dict[split].map(
            preprocessor.speech_file_to_array_fn
        )
        dataset_dict[split] = dataset_dict[split].map(preprocessor.augment_dataset)

        dataset_dict[split] = dataset_dict[split].filter(
            lambda example: example["audio"] is not None
            and example["sentence"] is not None
        )

else:
    dataset_dict = load_data_splits()


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-medium", language="Swahili", task="transcribe", model_max_length=225
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium", language="Swahili", task="transcribe", model_max_length=225
)


def fix_sentence(sentence):
    transcription = sentence

    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]

    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."
    transcription = (
        transcription[:-1].translate(str.maketrans("", "", string.punctuation))
        + transcription[-1]
    )
    return transcription


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    # audio = examples["audio"]
    audio = examples.get("audio")
    examples["input_features"] = None
    examples["labels"] = None
    if audio is not None:
        examples["input_features"] = feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        sentences = fix_sentence(examples["sentence"])

        # encode target text to label ids
        examples["labels"] = tokenizer(
            sentences, max_length=225, truncation=True
        ).input_ids
    return examples


for k in dataset_dict:
    dataset_dict[k] = (
        dataset_dict[k]
        .map(
            prepare_dataset,
        )
        .with_format("torch")
    )

# dataset_dict["train"] = dataset_dict["train"].shuffle(buffer_size=500)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [
            {
                "input_ids": self.processor.tokenizer.truncate_sequences(
                    feature["labels"]
                )[0]
            }
            for feature in features
        ]
        # pad the labels to max length

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

# evaluate with the 'normalised' WER
do_normalize_eval = True


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, normalize=do_normalize_eval
    )
    label_str = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, normalize=do_normalize_eval
    )

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium", use_cache=False
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False


# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


def load_samples_dataset(dataset, num_samples=10):
    samples = []
    for i, item in enumerate(dataset):
        samples.append(item)
        if i == (num_samples - 1):
            break
    sample_dataset = Dataset.from_list(samples)
    return sample_dataset


def compute_spectrograms(example):
    waveform = example["audio"]
    specs = feature_extractor(
        waveform, sampling_rate=16000, padding="do_not_pad"
    ).input_features[0]
    return {"spectrogram": specs}


def record_to_html(sample_record):
    audio_array = np.array(sample_record["audio"])
    audio_sr = 16000
    # print(sample_record.keys())
    audio_duration = 5
    audio_spectrogram = np.array(sample_record["spectrogram"])

    bounds = (0, 0, audio_duration, audio_spectrogram.max())

    waveform_int = np.int16(audio_array * 32767)

    hv_audio = pn.pane.Audio(
        waveform_int, sample_rate=audio_sr, name="Audio", throttle=500
    )

    slider = pn.widgets.FloatSlider(end=audio_duration, visible=False, step=0.001)
    line_audio = hv.VLine(0).opts(color="black")
    line_spec = hv.VLine(0).opts(color="red")

    slider.jslink(hv_audio, value="time", bidirectional=True)
    slider.jslink(line_audio, value="glyph.location")
    slider.jslink(line_spec, value="glyph.location")

    time = np.linspace(0, audio_duration, num=len(audio_array))
    line_plot_hv = (
        hv.Curve((time, audio_array), ["Time (s)", "amplitude"]).opts(
            width=500, height=150, axiswise=True
        )
        * line_audio
    )

    hv_spec_gram = (
        hv.Image(
            audio_spectrogram, bounds=(bounds), kdims=["Time (s)", "Frequency (hz)"]
        ).opts(width=500, height=150, labelled=[], axiswise=True, color_levels=512)
        * line_spec
    )

    combined = pn.Row(hv_audio, hv_spec_gram, line_plot_hv, slider)
    audio_html = StringIO()
    combined.save(audio_html)
    return audio_html


def dataset_to_records(dataset):
    records = []
    for item in dataset:
        record = {}
        item.keys()
        record["audio_with_spec"] = wandb.Html(record_to_html(item))
        record["sentence"] = item["sentence"]
        record["length"] = 5
        records.append(record)
    records = pd.DataFrame(records)
    return records


def decode_predictions(trainer, predictions):
    pred_ids = predictions.predictions
    pred_str = trainer.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )
    return pred_str


def compute_measures(predictions, labels):
    measures = [
        jiwer.compute_measures(
            ls,
            ps,
        )
        for ps, ls in zip(predictions, labels)
    ]
    measures_df = pd.DataFrame(measures)[
        ["wer", "hits", "substitutions", "deletions", "insertions"]
    ]
    return measures_df


class WandbProgressResultsCallback(WandbCallback):
    def __init__(self, trainer, sample_dataset):
        super().__init__()
        self.trainer = trainer
        self.sample_dataset = sample_dataset
        self.records_df = dataset_to_records(sample_dataset)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs)
        predictions = trainer.predict(self.sample_dataset)
        predictions = decode_predictions(self.trainer, predictions)
        measures_df = compute_measures(
            predictions, self.records_df["sentence"].tolist()
        )
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"sample_predictions": records_table})

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(
                    name=f"model-{self._wandb.run.id}", type="model", metadata=metadata
                )
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)


training_args = Seq2SeqTrainingArguments(
    output_dir=f"./whisper-medium-{language}",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    save_total_limit=4,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    #     fp16_full_eval=True,
    optim="adamw_bnb_8bit",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    save_strategy="epoch",
    eval_steps=500,
    logging_steps=25,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    remove_unused_columns=False,
    ignore_data_skip=True,
    run_name=f"whisper-medium-{language}",
)

samples_dataset = load_samples_dataset(dataset_dict["valid"]).map(compute_spectrograms)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["vaid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[ShuffleCallback(), EarlyStoppingCallback(early_stopping_patience=2)],
)

progress_callback = WandbProgressResultsCallback(trainer, samples_dataset)

trainer.add_callback(progress_callback)

trainer.train()
