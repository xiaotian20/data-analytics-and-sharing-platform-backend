from django.db import models
from django.contrib.auth.models import User

class Agency(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    accessible_data = models.ManyToManyField('Data', blank=True, related_name='agencies')

class Data(models.Model):
    agency = models.ForeignKey(Agency, on_delete=models.CASCADE)
    data = models.FileField(upload_to='datasets/')
    public = models.BooleanField(default=False)

