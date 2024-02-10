from django.db import models


class AudioGeneration(models.Model):
    audio=models.FileField(upload_to='generateAudio/',blank=True,null=True)
