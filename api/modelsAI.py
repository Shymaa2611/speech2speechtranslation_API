from pydub import AudioSegment
import os
import torch
from django.core.files import File
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from transformers import MarianTokenizer, MarianMTModel
#from model.utils.generation import SAMPLE_RATE, generate_audio, preload_models
#from scipy.io.wavfile import write as write_wav
#import shutil
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from .models import Audio_segment
from django.core.files.base import ContentFile

speech_regions=[{'start': 0, 'end': 14.456706281833617},
 {'start': 24.69439728353141, 'end': 31.6044142614601},
 {'start': 42.50424448217318, 'end': 42.89473684210526},]

non_speech=speech_regions=[{'start': 0, 'end': 14.456706281833617},
 {'start': 24.69439728353141, 'end': 31.6044142614601},
 {'start': 42.50424448217318, 'end': 42.89473684210526},]

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

def convert_segment_to_speech():
    speech_segments = Audio_segment.objects.filter(type="speech")
    texts = []
    for segment in speech_segments:
        audio_data = segment.audio.read()
        text = speech_to_text_process(audio_data)
        texts.append(text)
    
    print(texts)

""""
def text_to_text_translation(text):
    mname = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_text

def text_to_speech(text,audio,audio_num):
   preload_models()
   audio_array = generate_audio(text, prompt=audio)
   os.makedirs("target_dir", exist_ok=True)
   audio_path = os.path.join("target_dir", f"audio_{audio_num}.wav")
   write_wav(audio_path, SAMPLE_RATE, audio_array)

def speech_construct():
    audio_segments = []
    folder_path = "target_dir"
    file_names = sorted(os.listdir(folder_path))  
    for file_name in file_names:
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            audio_segment = AudioSegment.from_wav(file_path)
            audio_segments.append(audio_segment)

    target_audio = sum(audio_segments)
    try:
      shutil.rmtree(folder_path)
      print(f"Folder '{folder_path}' removed successfully.")
    except OSError as e:
      print(f"Error: {folder_path} : {e.strerror}")
    os.makedirs("target_dir", exist_ok=True)
    output_path = os.path.join(folder_path, "target.wav")
    target_audio.export(output_path, format="wav")


    


#source  => english speech
#target  => arabic speeech

def speech_to_speech_translation_en_ar(audio_url):
    output_dir = "outputSegment"
    segments = split_audio_segments(audio_url)
    for i, segment in enumerate(segments):
       wav_file = f"segment_{i}.wav"
       audio_url = os.path.join(output_dir, wav_file)
       en_text = speech_to_text_process(audio_url)
       translated_text=text_to_text_translation(en_text)
       translated_text = " ".join(translated_text)
       text_to_speech(translated_text,audio_url,i)
    speech_construct() 
    try:
      shutil.rmtree(output_dir)
      print(f"Folder '{output_dir}' removed successfully.")
    except OSError as e:
      print(f"Error: {output_dir} : {e.strerror}")
    
"""
    

#if __name__=="__main__":
#    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    #segments=split_audio_segments(audio_url)
    #text=speech_to_text_process(audio_url)
    #segments=split_audio_segments(audio_url)
    # source text =>>>>> english
    #speech_to_speech_translation_en_ar(audio_url)
    # target text =>>>>> arabic
    #target_text=text_to_text_translation(source_text)
  #  split_audio_segments(audio_url)

    
