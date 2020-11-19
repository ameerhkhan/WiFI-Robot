from django.urls import path
from Objector import views

# root folder indicated via ""

urlpatterns = [
    path("", views.objector_video, name='objector_video'),
]

