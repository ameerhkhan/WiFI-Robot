# Generated by Django 3.1.2 on 2020-11-12 15:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Objector', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='detector',
            name='accuracy_score',
            field=models.IntegerField(),
        ),
    ]
