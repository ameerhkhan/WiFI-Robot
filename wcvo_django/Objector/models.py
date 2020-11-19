from django.db import models

# Create your models here.
class Detector(models.Model):
    detected_object = models.CharField(max_length=100)
    accuracy_score = models.IntegerField()
    bounding_box = models.CharField(max_length=20)

    def __str__(self):
        return self.detected_object