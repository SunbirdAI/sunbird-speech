from speaker_diarization import AudioFile, TranscribeVoiceActivity
transcription_segments = []
transcribe_only_voice_activity = TranscribeVoiceActivity(target_lang="ach") #Pick from available eng, ach, lug, lgg, nyn, teo

for file_to_transcribe in os.listdir("/content/acholi_files"):
    audio_file = AudioFile(f"/content/acholi_files/{file_to_transcribe}")
    transcribe_only_voice_activity.transcribe(audio_file)
    for segment in audio_file.segments["combined"]:
        print(segment.transcript)
    transcription_segments.append(audio_file)

