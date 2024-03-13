from rest_framework import serializers
from .models import AudioGeneration,Audio_segment
class AudioGenerationSerializers(serializers.ModelSerializer):
    class Meta:
        model=AudioGeneration
        fields='__all__'

class AudioSegmentSerializers(serializers.ModelSerializer):
    class Meta:
        model=Audio_segment
        fields='__all__'


 