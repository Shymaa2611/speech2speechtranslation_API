from pydub import AudioSegment
import os
import torch
from django.core.files import File
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MarianTokenizer, MarianMTModel
from vallex.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import shutil
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from .models import Audio_segment,AudioGeneration
from django.core.files.base import ContentFile
from io import BytesIO



def audio_speech_nonspeech_detection(audio_url):
    model = Model.from_pretrained(
     "pyannote/segmentation-3.0", 
      use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
      "min_duration_on": 0.0,
      "min_duration_off": 0.0
     }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_url)
    speaker_regions=[]
    for turn, _,speaker in vad.itertracks(yield_label=True):
         speaker_regions.append({"start":turn.start,"end":turn.end})
    sound = AudioSegment.from_wav(audio_url)
    speaker_regions.sort(key=lambda x: x['start'])
    non_speech_regions = []
    for i in range(1, len(speaker_regions)):
        start = speaker_regions[i-1]['end'] 
        end = speaker_regions[i]['start']   
        if end > start:
            non_speech_regions.append({'start': start, 'end': end})
    first_speech_start = speaker_regions[0]['start']
    if first_speech_start > 0:
          non_speech_regions.insert(0, {'start': 0, 'end': first_speech_start})
    last_speech_end = speaker_regions[-1]['end']
    total_audio_duration = len(sound)  
    if last_speech_end < total_audio_duration:
            non_speech_regions.append({'start': last_speech_end, 'end': total_audio_duration})
    return speaker_regions,non_speech_regions

 
def split_audio_segments(audio_url):
    sound = AudioSegment.from_wav(audio_url)
    speech_segments, non_speech_segment = audio_speech_nonspeech_detection(audio_url)
    
    # Process speech segments
    for i, speech_segment in enumerate(speech_segments):
        start = int(speech_segment['start'] * 1000)  
        end = int(speech_segment['end'] * 1000)  
        segment = sound[start:end]
        audio_segment = Audio_segment(
            start_time=start/1000,
            end_time=end/1000,
            type="speech"
        )
        temp_file_path = f"temp_segment_{i}.wav"
        segment.export(temp_file_path, format="wav")
        with open(temp_file_path, "rb") as f:
            audio_segment.audio.save(f"speech_{i}.wav", File(f))
    
        os.remove(temp_file_path)
        audio_segment.save()
    
    # Process non-speech segments 
    for i, non_speech_segment in enumerate(non_speech_segment):
        start = int(non_speech_segment['start'] * 1000)  
        end = int(non_speech_segment['end'] * 1000)  
        segment = sound[start:end]
        audio_segment = Audio_segment(
            start_time=start/1000,
            end_time=end/1000,
        )
        temp_file_path = f"temp_segment_{i}.wav"
        segment.export(temp_file_path, format="wav")
        with open(temp_file_path, "rb") as f:
            audio_segment.audio.save(f"non_speech_{i}.wav", File(f))
        os.remove(temp_file_path)
        audio_segment.save()

def speech_to_text_process(segment):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
           model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
     "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      max_new_tokens=128,
      chunk_length_s=30,
      batch_size=16,
      return_timestamps=True,
      torch_dtype=torch_dtype,
      device=device,
    )
    result = pipe(segment)
    return result["text"]

""" def convert_segment_to_speech():
    speech_segments = Audio_segment.objects.filter(type="speech")
    texts = []
    for segment in speech_segments:
        audio_data = segment.audio.read()
        text = speech_to_text_process(audio_data)
        texts.append(text)
    
    return texts
 """

def text_to_text_translation(text):
    mname = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_text

def text_to_speech(segment_id,target_text, audio_prompt):
    preload_models()
    segment = Audio_segment.objects.get(id=segment_id)
    audio_array = generate_audio(target_text,audio_prompt)
    segment.audio.delete(save=False)
    audio_data = BytesIO(audio_array.tobytes())
    segment.audio.save(f"new_audio_{segment_id}.wav", File(audio_data))


def construct_audio():
    segments = Audio_segment.objects.all().order_by('start_time')
    audio_files = [AudioSegment.from_file(segment.audio.path) for segment in segments]
    target_audio = sum(audio_files)
    target_audio_path = "target_audio.wav"
    target_audio.export(target_audio_path, format="wav")
    audio_generation = AudioGeneration.objects.create(audio=target_audio_path)
    Audio_segment.objects.all().delete()

#source  => english speech
#target  => arabic speeech
def speech_to_speech_translation_en_ar(audio_url):
    split_audio_segments(audio_url)
    speech_segments = Audio_segment.objects.filter(type="speech")
    for segment in speech_segments:
        audio_data = segment.audio.read()
        text = speech_to_text_process(audio_data)
        target_text=text_to_text_translation(text)
        segment_id = segment.id
        audio_file_path = segment.audio.path
        text_to_speech(segment_id,target_text,audio_file_path)

    construct_audio()
   

