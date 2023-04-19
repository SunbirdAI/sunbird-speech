import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration


MODEL_PATH = "/home/mila/a/akeraben/scratch/akera/code/sunbird-speech/speech-to-text/whisper-small-sw/checkpoint-1000/"
AUDIO_PATH = "/home/mila/a/akeraben/scratch/akera/data/cv-corpus-10.0-2022-07-04/lg/clips/common_voice_lg_23703609.mp3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

y, _ = librosa.load(AUDIO_PATH, sr=16000)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted = torch.argmax(logits, dim=-1)
predicted[predicted == -100] = processor.tokenizer.pad_token_id
transcription = processor.tokenizer.batch_decode(predicted)[0]

print(transcription)
