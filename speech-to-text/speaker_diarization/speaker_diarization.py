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


class PipelineManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineManager, cls).__new__(cls)
            cls._pipeline = None
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def load_pipeline(self, pipeline_type, model_name, hyper_parameters):
        if self._pipeline is not None:
            # Optionally, add logic to handle switching between different pipelines
            self.unload_pipeline()

        if pipeline_type == "VoiceActivityDetection":
            self._pipeline = VoiceActivityDetection(segmentation=model_name)
        elif pipeline_type == "OverlappedSpeechDetection":
            self._pipeline = OverlappedSpeechDetection(segmentation=model_name)
        elif pipeline_type == "Diarization":
            self._pipeline = Pipeline.from_pretrained(model_name)

        self._pipeline.to(self._device)
        self._pipeline.instantiate(hyper_parameters)

    def unload_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.to(torch.device("cpu"))
            del self._pipeline
            self._pipeline = None

    def process_audio(self, audio_path):
        if self._pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        return self._pipeline(audio_path)

class AudioFile:
    """
    A class to represent a single audiofile with multiple segments.

    This class is used to store the segments of an audio file that can be used to generate transcripts and

    Attributes:
        path (str): Path to the audio file
        segments List[Segment]: A list of segments in an audio file.
    """

    def __init__(self, path):
        self.path = path
        self.segments = {}

    #def load_file(self):
    #    self.segments = open(self.path)#FIXME add functionality



class Segment:
    """
    A class to represent a single segment of audio with a start time, end time, and speaker label.

    This class is typically used to encapsulate the information about a segment of audio that
    has been identified during a speaker diarization process, including the time the segment
    starts, when it ends, and which speaker is speaking.

    Attributes:
        start (float): The start time of the audio segment in seconds.
        end (float): The end time of the audio segment in seconds.
        speaker (str, optional): The label of the speaker for this audio segment. Defaults to None.
    """

    def __init__(self, start, end, audio_file_path=None, speaker=None, voice_activity=None, overlap=None, transcript=None, audio=None ):
        """
        Initializes a new instance of the Segment class.

        Args:
            start (float): The start time of the audio segment in seconds.
            end (float): The end time of the audio segment in seconds.
            speaker (str, optional): The label of the speaker for this segment. If not specified,
                                     the speaker attribute is set to None.
        """
        self.start = start
        self.end = end
        self.speaker = speaker
        self.voice_activity = voice_activity
        self.overlap = overlap
        self.transcript = transcript
        self._audio = None
        self.audio_file_path = audio_file_path

    @property
    def audio(self):
        if self._audio is None:
            # Load audio segment lazily
            self._load_audio()
        return self._audio

    def _load_audio(self):
        # Load the specific segment of the audio file
        self._audio, _ = librosa.load(self.audio_file_path, sr=None, mono=True, offset=self.start, duration=self.end - self.start)

    def clear_audio(self):
        # Clear the loaded audio to free up memory
        self._audio = None




class Transcribe:
    '''
    This class is the main operational class that applies transcriptions on AudioFile instances.
    The main inputs and outputs to this class are AudioFile.
    '''

    def __init__(self, model_id="Sunbird/sunbird-mms",device = "cuda", target_lang="lug"):
        print(model_id)
        self.pipe = pipeline(model=model_id, device=device)
        self.pipe.tokenizer.set_target_lang(target_lang)
        self.pipe.model.load_adapter(target_lang)


    ## Combine vad with existing segments if existing
            #combination strategy split vad areas by overlap if vad exists. 1-10 vad 2-3 overlap 1-2 vad + 2-3 vad overlap + 3-10 vad
            #combination strategy for speakers is to split each vad area by speaker. speaker overwrites vad.
            ## O(N^2) double loop
            ## Sorted loops with indices, use a dict with operation name as


    ##Implement some state, read/write functionality in case restarts are needed
    def transcribe(self, audio_file):
        segments = audio_file.segments
        ### Check to see if audiofile has been segmented or not
        if len(audio_file.segments) > 0:
            segments = audio_file.segments
        ## Open file from start to finish as a single segment
        else:
            raise NotImplementedError("Support for directly transcribing files not implemented. Must use a subclass with a Diarization Mixin first")

        if "vad" in segments:


            if "speakers" in segments:
                segments_with_no_overlap = self.add_overlapped_speaker(segments["speakers"], segments["overlap"])

            elif "overlap" in segments:
                segments_with_no_overlap = self.remove_overlapped(segments["vad"], segments["overlap"])

            else:
                segments_with_no_overlap = segments["vad"]


        elif "overlap" in segments:
            if "speakers" in segments:
                segments_with_no_overlap = self.add_overlapped_speaker(segments["speakers"], segments["overlap"])

        elif "speakers" in segments:
            segments_with_no_overlap = segments["speakers"]

        else:
            raise NotImplementedError("Where are the diarization mixins?")

        audio_file.segments["combined"] = segments_with_no_overlap

        for segment in audio_file.segments["combined"]:
            segment_transcription = self.transcribe_audio(segment.audio)
            segment.transcript = segment_transcription

    @staticmethod
    def add_overlapped_speaker(overlap_segments, speaker_segments):
        combined_segments = []

        for speaker_segment in speaker_segments:
            overlaps = [o for o in overlap_segments if o.start < speaker_segment.end and o.end > speaker_segment.start]

            if not overlaps:
                combined_segments.append(speaker_segment)
                continue

            for overlap in overlaps:
                # Segment before overlap
                if speaker_segment.start < overlap.start:
                    pre_overlap = Segment(start=speaker_segment.start, end=overlap.start,
                                          speaker=speaker_segment.speaker,
                                        voice_activity=speaker_segment.voice_activity,
                                        overlap=False,
                                        transcript=speaker_segment.transcript,
                                        audio_file_path=speaker_segment.audio_file_path)
                    combined_segments.append(pre_overlap)

                # Overlap segment
                overlap_segment = Segment(start=max(speaker_segment.start, overlap.start),
                                        end=min(speaker_segment.end, overlap.end),
                                        speaker="OVERLAP",  # Overlapping segments marked as "OVERLAP"
                                        voice_activity=speaker_segment.voice_activity,
                                        overlap=True,
                                        transcript="*** OVERLAPPING SPEECH ***",  # Overlaps might not have a clear transcript
                                        audio_file_path=speaker_segment.audio_file_path)
                combined_segments.append(overlap_segment)

                # Segment after overlap
                if speaker_segment.end > overlap.end:
                    post_overlap = Segment(start=overlap.end, end=speaker_segment.end,
                                           speaker=speaker_segment.speaker,
                                            voice_activity=speaker_segment.voice_activity,
                                            overlap=False,
                                            transcript=speaker_segment.transcript,
                                            audio_file_path=speaker_segment.audio_file_path)
                    combined_segments.append(post_overlap)

        return combined_segments


    @staticmethod
    def remove_overlapped(vad_segments, overlap_segments):
        # Assuming vad_segments and overlap_segments are lists of Segment objects
        adjusted_segments = []

        for vad_segment in vad_segments:
            overlap_ranges = [
                (max(overlap_segment.start, vad_segment.start), min(overlap_segment.end, vad_segment.end))
                for overlap_segment in overlap_segments
                if overlap_segment.start < vad_segment.end and overlap_segment.end > vad_segment.start
            ]

            if not overlap_ranges:
                adjusted_segments.append(vad_segment)  # No overlap, keep the segment as is
                continue

            overlap_ranges.sort()
            current_start = vad_segment.start

            # Split the vad_segment around each overlap
            for start, end in overlap_ranges:
                if current_start < start:
                    adjusted_segments.append(Segment(current_start, start,
                                                     speaker=vad_segment.speaker,
                                                    voice_activity=vad_segment.voice_activity,
                                                    overlap=False,  # Assuming no overlap in this segment
                                                    transcript=vad_segment.transcript,
                                                    audio_file_path=vad_segment.audio_file_path))  # Before overlap
                adjusted_segments.append(Segment(start, end,
                                                 speaker="OVERLAP",  # You might want to adjust this
                                                voice_activity=vad_segment.voice_activity,
                                                overlap=True,
                                                transcript=None,  # Overlaps might not have a clear transcript
                                                audio_file_path=vad_segment.audio_file_path))  # Overlap
                current_start = end

            if current_start < vad_segment.end:
                adjusted_segments.append(Segment(current_start, vad_segment.end,
                                                 speaker=vad_segment.speaker,
                                                voice_activity=vad_segment.voice_activity,
                                                overlap=False,
                                                transcript=vad_segment.transcript,
                                                audio_file_path=vad_segment.audio_file_path))  # After overlap

        return adjusted_segments

    def transcribe_audio(self,input_file,
                     chunk_length_s=10,
                     stride_length_s=(4, 2),
                     return_timestamps="word"):
        #FIXME does this need to be through a file path?
        output = self.pipe(input_file, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s,return_timestamps="word")
        #TODO what's the output shape and how to shape it to segments?
        return output

# class Mixin1:
#     def method(self):
#         print("Mixin1")
#         super().method()

# class Mixin2:
#     def method(self):
#         print("Mixin2")
#         super().method()

# class BaseClass:
#     def method(self):
#         print("BaseClass")

# class MyClass(Mixin1, Mixin2, BaseClass):
#     pass

## Create Singleton classes with RAM + GPU garbage collection for using pyannote models
class VoiceActivityMixin:

    HYPER_PARAMETERS = {
    # onset/offset activaton thresholds
    #"onset": 0.5, "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }

    def transcribe(self, audio_file):
        ## Do stuff then call parent class
        ## Detect voice activity
        pipeline_manager = PipelineManager()
        pipeline_manager.load_pipeline("VoiceActivityDetection", "pyannote/segmentation-3.0", self.HYPER_PARAMETERS)
        vad = pipeline_manager.process_audio(audio_file.path)

        for segment_info, _, _ in vad.itertracks(yield_label=True):
            segment = Segment(start=segment_info.start, end=segment_info.end, voice_activity=True, audio_file_path=audio_file.path)
            try:
                audio_file.segments["vad"].append(segment)
            except:
                audio_file.segments["vad"] = [segment]


        ## call parent class transcribe
        super(VoiceActivityMixin,self).transcribe(audio_file)

class OverlapMixin:

    '''
    This mixin can be used before other diarization mixins to remove overlapped bits or to mark them as potentially
    problematic
    '''

    HYPER_PARAMETERS = {
    # onset/offset activaton thresholds
    #"onset": 0.5, "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }

    def transcribe(self, audio_file):
        ## Do stuff then call parent class
        ## Detect voice activity
        pipeline_manager = PipelineManager()
        pipeline_manager.load_pipeline("OverlappedSpeechDetection", "pyannote/segmentation-3.0", self.HYPER_PARAMETERS)
        overlap = pipeline_manager.process_audio(audio_file.path)

        for segment_info, _, _ in overlap.itertracks(yield_label=True):
            segment = Segment(start=segment_info.start, end=segment_info.end, overlap=True, audio_file_path=audio_file.path)
            try:
                audio_file.segments["overlap"].append(segment)
            except:
                audio_file.segments["overlap"] = [segment]



        ## call parent class transcribe
        super(OverlapMixin, self).transcribe(audio_file)

class DiarizationMixin:
    def transcribe(self, audio_file):
        ## Do stuff then call parent class
        ## Detect voice activity

        pipeline_manager = PipelineManager()
        pipeline_manager.load_pipeline("Diarization", "pyannote/segmentation-3.0", self.HYPER_PARAMETERS)
        overlap = pipeline_manager.process_audio(audio_file.path)

        diarization = pipeline_manager.process_audio(audio_file.path)

        for segment_info, speaker_id, speaker_name in overlap.itertracks(yield_label=True):
            segment = Segment(start=segment_info.start, end=segment_info.end, speaker=speaker_name, audio_file_path=audio_file.path)
            try:
                audio_file.segments["speakers"].append(segment)
            except:
                audio_file.segments["speakers"] = [segment]
        ## Combine with existing segments if existing



        ## call parent class transcribe
        super(DiarizationMixin, self).transcribe(audio_file)

class TranscribeVoiceActivity(VoiceActivityMixin, Transcribe):
    pass


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
