from django.urls import path, include
from . import views

urlpatterns = [
    path('upload/', views.upload),
    path('recognize/', views.recognize),
    path('upload_seperate/', views.upload_seperate),
    path('seperate/', views.seperate),
    path('download/', views.download),
]
