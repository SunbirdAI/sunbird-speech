# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenSLR Dataset"""

from __future__ import absolute_import, division, print_function

import os
import re
from pathlib import Path

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_DATA_URL = "https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition/data"

_CITATION = """\
"""

_DESCRIPTION = """\
Zindi dataset
"""

_HOMEPAGE = "https://zindi.africa/"

_LICENSE = ""

_RESOURCES = {
    "zindi": {
        "Language": "Luganda",
        "LongName": "Speech dataset for Luganda",
        "Category": "Speech",
    },
}


class ZindiConfig(datasets.BuilderConfig):
    """BuilderConfig for Zindi."""

    def __init__(self, name, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.language = kwargs.pop("language", None)
        self.long_name = kwargs.pop("long_name", None)
        self.category = kwargs.pop("category", None)
        super(ZindiConfig, self).__init__(name=name, **kwargs)


class Zindi(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        ZindiConfig(
            name="lg",
            version=VERSION,
            description="Zindi dataset",
        ),
        ZindiConfig(
            name="lg_old",
            version=VERSION,
            description="Zindi dataset",
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "sentence": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                AutomaticSpeechRecognition(audio_file_path_column="path", transcription_column="sentence")
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        print(f"manual_dir: {dl_manager.manual_dir}")
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        print(f"datadir: {data_dir}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('id_liputan6', "
                "'canonical', data_dir=...)`. Manual download instructions:\n{}".format(
                    data_dir, self.manual_download_instructions
                )
            )
        if self.config.name == "lg":
            split_generators = [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "path_to_index": os.path.join(data_dir, f"{data_dir}/Test.csv"),
                        "path_to_data": os.path.join(data_dir, f"{data_dir}/content/test_dataset"),
                        "split": "test"
                    },
                ),
            ]
        else:
            split_generators = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "path_to_index": os.path.join(data_dir, f"{data_dir}/Train.csv"),
                        "path_to_data": os.path.join(data_dir, f"{data_dir}/validated_dataset"),
                        "split": "train"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "path_to_index": os.path.join(data_dir, f"{data_dir}/Test.csv"),
                        "path_to_data": os.path.join(data_dir, f"{data_dir}/validated_dataset"),
                        "split": "test"
                    },
                ),
            ]
        return split_generators

    def _generate_examples(self, path_to_index, path_to_data, split):
        """Yields examples."""
        counter = -1
        if self.config.name in ["lg"]:
            sentence_index = {}
            with open(path_to_index, encoding="utf-8") as f:
                lines = f.readlines()
                for id_, line in enumerate(lines):
                    if id_ == 0:
                        continue
                    field_values = re.split(r",", line.strip())
                    if split == "train":
                        clip_id, client_id, up_votes, down_votes, age, gender, sentence = field_values
                        sentence_index[clip_id] = sentence
                    else:
                        clip_id, client_id, up_votes, down_votes = field_values
                        sentence_index[clip_id] = ""
            for path_to_soundfile in sorted(Path(path_to_data).rglob("*.mp3")):
                filename = path_to_soundfile.stem
                if filename not in sentence_index:
                    continue
                path = str(path_to_soundfile.resolve())
                sentence = sentence_index[filename]
                counter += 1
                yield counter, {"path": path, "sentence": sentence}
        else:
            pass
