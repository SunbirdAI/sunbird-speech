import os
import glob
import pandas as pd
from datasets import load_dataset, Audio, Dataset
import librosa

class SaltSpeechDataset:
    """
    A class used to represent the SaltSpeechDataset

    ...

    Attributes
    ----------
    datasets : dict
        a dictionary that holds the datasets for each language

    Methods
    -------
    download_files():
        Downloads the audio files and transcripts for all languages.

    extract_audio_files():
        Extracts all downloaded .zip files in the 'data' directory.

    load_and_split_dataset(audio_dir, csv_path):
        Loads and splits the dataset for a given language.

    prepare_datasets():
        Prepares the datasets for all languages, ready for use.
    """
    # Define constants
    AUDIO_URL = "https://storage.googleapis.com/sb-public-datasets/asr-speech/{}-validated.zip"
    TRANSCRIPT_URL = "https://storage.googleapis.com/sb-public-datasets/asr-speech/Prompt-{}.csv"
    LANGUAGES = [("acholi", "Acholi"), ("ateso", "Ateso"),
                 ("luganda", "Luganda"), ("lugbara", "Lugbara"), ("runyankole", "Runyankole")]

    def __init__(self):
        """Initializes the SaltSpeechDataset with an empty dictionary for the datasets."""
        self.datasets = {}

    def download_files(self):
        """
        Downloads the audio files and transcripts for all languages.

        Files are downloaded from defined URLs and stored in the 'data' directory.
        Only downloads files if they do not already exist.
        """
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Download audio files and transcripts for all languages
        for lang_audio, lang_trans in self.LANGUAGES:
            audio_file_path = f"data/{lang_audio}-validated.zip"
            transcript_file_path = f"data/Prompt-{lang_trans}.csv"

            # Only download the files if they do not already exist
            if not os.path.exists(audio_file_path):
                os.system(f"wget {self.AUDIO_URL.format(lang_audio)} -P data")
            if not os.path.exists(transcript_file_path):
                os.system(f"wget {self.TRANSCRIPT_URL.format(lang_trans)} -P data")


    def extract_audio_files(self):
        """
        Extracts all downloaded .zip files in the 'data' directory.

        Audio files are extracted and kept in the 'data' directory.
        """
        # Unzip all downloaded files in 'data' directory
        os.system("unzip 'data/*.zip' -d data")

    def load_and_split_dataset(self, audio_dir, csv_path):
        """
        Loads and splits the dataset for a given language.

        Transcripts are read from a CSV file and matched with corresponding audio files.
        The dataset is split into train, test, and validation subsets.

        Parameters:
        audio_dir (str): The directory where audio files are stored
        csv_path (str): The path of the CSV file containing transcripts

        Returns:
        dict: A dictionary with keys 'train', 'test', 'validation' and values being the respective subsets
        """
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        def get_audio_paths(row):
            key = os.path.join(audio_dir, str(row['Key']))
            audio_files = glob.glob(os.path.join(key + "/*.ogg"))
            return audio_files

        # Apply the function to the DataFrame to get the audio paths
        df['audio_paths'] = df.apply(get_audio_paths, axis=1)

        # Convert the DataFrame to a HuggingFace Dataset
        dataset = Dataset.from_pandas(df)

        # Split the dataset
        train_dataset = dataset.filter(lambda example: example['split'] == 'train')
        test_dataset = dataset.filter(lambda example: example['split'] == 'test')
        val_dataset = dataset.filter(lambda example: example['split'] == 'val')

        return {"train": train_dataset, "test": test_dataset, "validation": val_dataset}

    def prepare_datasets(self):
        """
        Prepares the datasets for all languages, ready for use.

        Downloads necessary files, extracts audio files, and loads and splits datasets for each language.
        """
        # Download necessary files
        self.download_files()

        # Extract audio files from zip
        self.extract_audio_files()

        # Load and split datasets for each language
        for lang_audio, lang_trans in self.LANGUAGES:
            audio_dir = f"data/{lang_audio}-validated"
            csv_path = f"data/Prompt-{lang_trans}.csv"
            self.datasets[lang_audio] = self.load_and_split_dataset(audio_dir, csv_path)
