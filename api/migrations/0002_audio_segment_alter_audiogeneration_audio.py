# Generated by Django 4.1.10 on 2024-03-13 08:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Audio_segment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_time', models.FloatField()),
                ('end_time', models.FloatField()),
                ('type', models.CharField(default='non-speech', max_length=50)),
                ('audio', models.FileField(blank=True, null=True, upload_to='segments/')),
            ],
        ),
        migrations.AlterField(
            model_name='audiogeneration',
            name='audio',
            field=models.FileField(blank=True, null=True, upload_to='target/'),
        ),
    ]
