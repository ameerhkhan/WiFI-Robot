# Generated by Django 3.1.2 on 2020-11-12 15:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Detector',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('detected_object', models.CharField(max_length=100)),
                ('accuracy_score', models.IntegerField(max_length=100)),
                ('bounding_box', models.CharField(max_length=20)),
            ],
        ),
    ]
