import os
import glob
import pandas as pd

import pickle

import torch
import librosa

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForCTC, AutoFeatureExtractor
from transformers import Wav2Vec2ProcessorWithLM, pipeline

from tqdm import tqdm 

from datasets import load_dataset, Audio, Dataset

from pyannote.audio.pipelines import OverlappedSpeechDetection
import time

import torch
from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
from pyannote.audio.pipelines import OverlappedSpeechDetection, VoiceActivityDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_audio(input_file_path):
    # Pre-trained pipelines/models
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")
    pipeline_vad = VoiceActivityDetection(segmentation="pyannote/segmentation-3.0")
    overlap_detection = OverlappedSpeechDetection(segmentation="pyannote/segmentation-3.0")
    HYPER_PARAMETERS = {
    # onset/offset activaton thresholds
    #"onset": 0.5, "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }


    # Send models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)
    pipeline_vad.to(device)
    overlap_detection.to(device)

    # Apply voice activity detection
    pipeline_vad.instantiate(HYPER_PARAMETERS)
    vad = pipeline_vad(input_file_path)

    # Apply speaker diarization
    diarization_pipeline.instantiate({})
    diarization = diarization_pipeline(input_file_path)

    # Apply overlap detection
    overlap_detection.instantiate(HYPER_PARAMETERS)
    overlap = overlap_detection(input_file_path)

    segments = {}

    # Extract segments for each speaker and note overlaps
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # This checks for overlap within the speaker segment
        is_overlap = False
        current_segment = Segment(turn.start, turn.end)
        for segment in overlap.get_timeline():
            if current_segment.intersects(segment):
                is_overlap = True
                break
        try:
            segments[f"speaker_{speaker}"].append({
                "segment": (turn.start, turn.end),
                "overlap": is_overlap
            })
        except:
            segments[f"speaker_{speaker}"] = [{
                "segment": (turn.start, turn.end),
                "overlap": is_overlap
            }]

    # Extracting audio arrays for each segment can be done using another library like librosa
    # Here we're just returning the segments for simplicity
    return segments

def extract_audio_segments(input_file_path, segments):
    # Load the entire audio file
    y, sr = librosa.load(input_file_path, sr=None)

    audio_segments = {}
    for speaker, speaker_segments in segments.items():
        audio_segments[speaker] = []
        for segment_info in speaker_segments:
            start_sample = librosa.time_to_samples(segment_info['segment'][0], sr=sr)
            end_sample = librosa.time_to_samples(segment_info['segment'][1], sr=sr)
            audio_array = y[start_sample:end_sample]
            audio_segments[speaker].append({
                'audio': audio_array,
                'segment': segment_info['segment'],
                'overlap': segment_info['overlap']
            })
    return audio_segments

def process_audio_sequential(input_file_path):
     # Pre-trained pipelines/models
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")
    pipeline_vad = VoiceActivityDetection(segmentation="pyannote/segmentation-3.0")
    overlap_detection = OverlappedSpeechDetection(segmentation="pyannote/segmentation-3.0")
    HYPER_PARAMETERS = {
    # onset/offset activaton thresholds
    #"onset": 0.5, "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }


    # Send models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)
    pipeline_vad.to(device)
    overlap_detection.to(device)

    # Apply voice activity detection
    pipeline_vad.instantiate(HYPER_PARAMETERS)
    vad = pipeline_vad(input_file_path)

    # Apply speaker diarization
    diarization_pipeline.instantiate({})
    diarization = diarization_pipeline(input_file_path)

    # Apply overlap detection
    overlap_detection.instantiate(HYPER_PARAMETERS)
    overlap = overlap_detection(input_file_path)

    sequential_segments = []

    # Extract segments for each speaker and note overlaps
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # This checks for overlap within the speaker segment
        is_overlap = False
        current_segment = Segment(turn.start, turn.end)
        for segment in overlap.get_timeline():
            if current_segment.intersects(segment):
                is_overlap = True
                break

        sequential_segments.append({
            "speaker": f"speaker_{speaker}",
            "segment": (turn.start, turn.end),
            "overlap": is_overlap
        })

    # Now, sort segments based on start time to ensure they are sequential
    sequential_segments.sort(key=lambda x: x['segment'][0])

    del diarization_pipeline
    del pipeline_vad
    del overlap_detection


    return sequential_segments

def extract_audio_segments_sequential(input_file_path, segments):
    # Load the entire audio file
    y, sr = librosa.load(input_file_path, sr=None)

    audio_segments = []
    for segment_info in segments:
        start_sample = librosa.time_to_samples(segment_info['segment'][0], sr=sr)
        end_sample = librosa.time_to_samples(segment_info['segment'][1], sr=sr)
        audio_array = y[start_sample:end_sample]
        audio_segments.append({
            'speaker': segment_info['speaker'],
            'audio': audio_array,
            'segment': segment_info['segment'],
            'overlap': segment_info['overlap']
        })

    return audio_segments



def write_segments_to_disk(segments, directory):
    """
    Write segments and their associated audio to disk.

    Parameters:
    - segments: Result from extract_audio_segments_sequential
    - directory: Directory where the segments will be stored
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, 'segments.pkl'), 'wb') as f:
        pickle.dump(segments, f)

def read_segments_from_disk(directory):
    """
    Read segments and their associated audio from disk.

    Parameters:
    - directory: Directory from where the segments will be read

    Returns:
    - segments: List of segments and their associated audio
    """

    with open(os.path.join(directory, 'segments.pkl'), 'rb') as f:
        segments = pickle.load(f)

    return segments


def transcribe_audio(input_file,
                     target_lang,
                     device,
                     model_id="Sunbird/sunbird-mms",
                     chunk_length_s=10,
                     stride_length_s=(4, 2),
                     return_timestamps="word"):
    """
    Transcribes audio from the input file using sunbird asr model.

    Args:
        input_file (str): Path to the audio file for transcription.
        target_lang (str): Target language for transcription.
            'ach' - Acholi
            'lug' - Luganda
            'teo' - Ateso
            'lgg' - Lugbara
        device (str or torch.device): Device for running the model (e.g., 'cpu', 'cuda').
        model_id (str, optional): ID of the asr model. Defaults to "Sunbird/sunbird-mms".
        chunk_length_s (int, optional): Length of audio chunks in seconds. Defaults to 5.

    Returns:
        dict: A dictionary containing the transcription result.
            Example: {'text': 'Transcribed text here.'}
    """


    pipe = pipeline(model=model_id, device=device)
    if model_id in ["Sunbird/sunbird-mms"]:
        pipe.tokenizer.set_target_lang(target_lang)
        pipe.model.load_adapter(target_lang)

    output = pipe(input_file, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s,return_timestamps="word")
    return output

def transcribe_segments(sequential_segments,
                     input_file,
                     target_lang,
                     device,
                     model_id="Sunbird/sunbird-mms",
                     chunk_length_s=10,
                     stride_length_s=(4, 2),
                     return_timestamps="word",
                     ignore_overlapping = False):
    transcriptions = []

    pipe = pipeline(model=model_id, device=device)

    for segment in tqdm(sequential_segments):
        if ignore_overlapping and segment["overlap"]:
            continue

        if model_id in ["Sunbird/sunbird-mms"]:
            pipe.tokenizer.set_target_lang(target_lang)
            pipe.model.load_adapter(target_lang)

        output = pipe(segment["audio"], chunk_length_s=chunk_length_s, stride_length_s=stride_length_s,return_timestamps="word")
        output["speaker"] = segment["speaker"]
        output["overlap"] = segment["overlap"]
        transcriptions.append(output)
    return transcriptions

def create_text_from_dicts(dict_list):
    """
    Constructs a combined text from a list of dictionaries using the 'text' key.

    Parameters:
    - dict_list: List of dictionaries each containing a 'text' key

    Returns:
    - Combined text string
    """

    texts = [d['speaker'] + ": " + d['text'] + "\n" for d in dict_list if 'text' in d]
    return ' '.join(texts)