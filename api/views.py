from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import AudioGeneration
from .serializers import AudioGenerationSerializers
from .modelsAI import speech_to_speech_translation_en_ar

@api_view(['GET'])
def get_audio(request):
    audio_url = request.GET.get('audio_url')
    try:
        speech_to_speech_translation_en_ar(audio_url)
        audio_generation = AudioGeneration.objects.filter(audio=audio_url).first()
        serializer =AudioGenerationSerializers(audio_generation)
        return Response(serializer.data)
    except AudioGeneration.DoesNotExist:
        return Response(status=404)

#audio_url=""
#speech_to_speech_translation_en_ar(audio_url)
