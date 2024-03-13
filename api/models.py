from django.db import models


class AudioGeneration(models.Model):
    audio=models.FileField(upload_to='target/',blank=True,null=True)


class Audio_segment(models.Model):
    start_time=models.FloatField()
    end_time=models.FloatField()
    type=models.CharField(max_length=50,default="non-speech")
    audio=models.FileField(upload_to='segments/',blank=True,null=True)
    #content=models.CharField(max_length=500,blank=True,null=True)
    class Meta:
        ordering = ['start_time']



