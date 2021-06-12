from django.urls import path, include
from . import views

urlpatterns = [
    path('noise_upload/', views.noise_upload),
    path('target_upload/', views.target_upload),
    path('simulate/', views.simulate),
    path('download/', views.download),
]