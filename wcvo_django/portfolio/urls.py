from django.urls import path
from portfolio import views

# root folder is indicated via ""
urlpatterns = [
    path("", views.project_index, name='project_index'),
    path("<int:pk>/", views.project_detail, name='project_detail'),
    # pass the PK to view the correct project.
]

