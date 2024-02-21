from pydub import AudioSegment
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MarianTokenizer, MarianMTModel
import numpy as np

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

def text_to_speech(text):
    pass

def speech_construct(segments):
    original_audio = segments[0]
    for segment in segments[1:]:
        original_audio += segment
    return original_audio

"""
source  => english speech
target  => arabic speeech
"""
def speech_to_speech_translation_en_ar(audio_url):
    output_dir = "outputSegments"
    speech2text=[]
    segments = split_audio_segments(audio_url)
    for i, segment in enumerate(segments):
       wav_file = f"segment_{i}.wav"
       audio_url = os.path.join(output_dir, wav_file)
       en_text = speech_to_text_process(audio_url)
       speech2text.append(en_text)
    text2textT=[] # contain arabic text
    for text in  speech2text:
        translated_text=text_to_text_translation(text)
        text2textT.append(translated_text)
    target_segments=[] #contain arabic speech segments
    for text in text2textT:
        segment=text_to_speech(text)
        target_segments.append(segment)
    target_audio=speech_construct(target_segments) 
    return target_audio


if __name__=="__main__":
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    #segments=split_audio_segments(audio_url)
    #text=speech_to_text_process(audio_url)
    #segments=split_audio_segments(audio_url)
    # source text =>>>>> english
    source_text=speech_to_text_process(audio_url)
    # target text =>>>>> arabic
    #target_text=text_to_text_translation(source_text)

    
