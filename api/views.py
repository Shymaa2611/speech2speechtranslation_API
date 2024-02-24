from django.shortcuts import render
from .models import AudioGeneration
from .serializers import AudioGenerationSerializers
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .modelsAI import speech_to_speech_translation_en_ar
        

@api_view(['GET','POST'])
def get_audio(request):

    if request.method == 'GET':
        data=AudioGeneration.objects.all()
        serializer = AudioGenerationSerializers(data, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        audio_url=request.data.get('audio_url')
        serializer=AudioGeneration(data=request.data)
        if serializer.is_valid():
            speech_to_speech_translation_en_ar(audio_url)
            #serializer.save(audio=target)
            return Response(serializer.data, status= status.HTTP_201_CREATED)
        return Response(serializer.errors, status= status.HTTP_400_BAD_REQUEST)


@api_view(['GET','PUT','DELETE'])
def get_audio_pk(request, pk):
    try:
        audio= AudioGeneration.objects.get(pk=pk)
    except AudioGeneration.DoesNotExists:
        return Response(status=status.HTTP_404_NOT_FOUND)
    if request.method == 'GET':
        serializer = AudioGenerationSerializers(audio)
        return Response(serializer.data)
        
    elif request.method == 'PUT':
        serializer = AudioGenerationSerializers(audio, data= request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status= status.HTTP_400_BAD_REQUEST)
    if request.method == 'DELETE':
        audio.delete()
        return Response(status= status.HTTP_204_NO_CONTENT)














 