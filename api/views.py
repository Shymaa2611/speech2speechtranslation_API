from django.shortcuts import render
from .models import AudioGeneration
from .serializers import AudioGenerationSerializers
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from pydub import AudioSegment
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MarianTokenizer, MarianMTModel

def split_audio_segments(audio_url, output_dir="outputSegments"):
    sound = AudioSegment.from_wav(audio_url)
    segment_duration = 15* 1000
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


#pip install transformers==4.3.0 sentencepiece==0.1.95 nltk==3.5 protobuf==3.15.3 torch#
def text_to_text_translation(request,text):
    mname = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_text

def text_to_speech(text):
    pass


def speech_construct(request,segments):
    original_audio = segments[0]
    for segment in segments[1:]:
        original_audio += segment
    return original_audio

def speech_to_speech_translation_en_ar(request,audio_url):
    segments=split_audio_segments(audio_url)
    speech2text=[]
    for segment in segments[0:]:
        en_text=speech_to_text_process(segment)
        speech2text.append(en_text)
    text2textT=[]
    for text in  speech2text:
        translated_text=text_to_text_translation(text)
        text2textT.append(translated_text)
    target_segments=[]
    for text in text2textT:
        segment=text_to_speech(text)
        target_segments.append(segments)
    target_audio=speech_construct(target_segments)
        



#======================== GET - POST ================#

@api_view(['GET','POST'])
def get_audio(request):
    # GET
    if request.method == 'GET':
        audio=AudioGeneration.objects.all()
        serializer = AudioGenerationSerializers(audio, many=True)
        return Response(serializer.data)
    # POST
    elif request.method == 'POST':
        audio=request.data.get('audio')
        serializer =AudioGenerationSerializers(data= request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status= status.HTTP_201_CREATED)
        return Response(serializer.data, status= status.HTTP_400_BAD_REQUEST)

# GET PUT DELETE
@api_view(['GET','PUT','DELETE'])
def get_audio_pk(request, pk):
    try:
        audio= AudioGeneration.objects.get(pk=pk)
    except AudioGeneration.DoesNotExists:
        return Response(status=status.HTTP_404_NOT_FOUND)
    # GET
    if request.method == 'GET':
        serializer = AudioGenerationSerializers(audio)
        return Response(serializer.data)
        
    # PUT
    elif request.method == 'PUT':
        serializer = AudioGenerationSerializers(audio, data= request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status= status.HTTP_400_BAD_REQUEST)
    # DELETE
    if request.method == 'DELETE':
        audio.delete()
        return Response(status= status.HTTP_204_NO_CONTENT)


   
















 