from django.urls import path
from .import views
urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home, name='home'),
    path('video_stream', views.video_stream, name='video_stream'),
]