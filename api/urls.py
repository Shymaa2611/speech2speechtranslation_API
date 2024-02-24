from django.urls import path
from . import views

urlpatterns = [
   path('generate_audio/', views.audio,name="generate_audio"),
   #path('generate_audio/<int:pk>/', views.get_audio_pk,name="generate_audio_pk"),
]
 