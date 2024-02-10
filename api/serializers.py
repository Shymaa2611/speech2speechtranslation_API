from rest_framework import serializers
from .models import AudioGeneration

class AudioGenerationSerializers(serializers.ModelSerializer):
    class Meta:
        model=AudioGeneration
        fields='__all__'


 