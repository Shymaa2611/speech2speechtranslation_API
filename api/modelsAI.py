from pydub import AudioSegment
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MarianTokenizer, MarianMTModel
import numpy as np
import wave
from model.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

def split_audio_segments(audio_url, output_dir="outputSegments"):
    sound = AudioSegment.from_wav(audio_url)
    segment_duration = 15*1000
    total_segments = len(sound) // segment_duration
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    segments = []
    for i in range(total_segments+1):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = sound[start_time:end_time]
        segments.append(segment)
        segment.export(os.path.join(output_dir, f"segment_{i}.wav"), format="wav")
    
    return segments

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

def text_to_text_translation(text):
    mname = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_text

def text_to_speech(text,audio):
    preload_models()
    audio_array = generate_audio(text, prompt=audio)
    write_wav(f"target_dir/{audio}.wav", SAMPLE_RATE, audio_array)

def speech_construct():
    audio_segments = []
    folder_path="target_dir"
    file_names = sorted(os.listdir(folder_path))  
    for file_name in file_names:
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            with wave.open(file_path, 'rb') as audio_file:
                audio_segments.append(audio_file.readframes(audio_file.getnframes()))

    target_audio = b"".join(audio_segments)
    with open("target_audio.wav", "wb") as audio_out_file:
         audio_out_file.write(target_audio)


    

"""
source  => english speech
target  => arabic speeech
"""
def speech_to_speech_translation_en_ar(audio_url):
    output_dir = "outputSegments"
    segments = split_audio_segments(audio_url)
    for i, segment in enumerate(segments):
       wav_file = f"segment_{i}.wav"
       audio_url = os.path.join(output_dir, wav_file)
       en_text = speech_to_text_process(audio_url)
       translated_text=text_to_text_translation(en_text)
       translated_text = " ".join(translated_text)
       text_to_speech(translated_text,audio_url)
    speech_construct() 
    


if __name__=="__main__":
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    #segments=split_audio_segments(audio_url)
    #text=speech_to_text_process(audio_url)
    #segments=split_audio_segments(audio_url)
    # source text =>>>>> english
    source_text=speech_to_text_process(audio_url)
    # target text =>>>>> arabic
    #target_text=text_to_text_translation(source_text)

    
