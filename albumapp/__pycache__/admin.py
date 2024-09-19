from django.contrib import admin
from albumapp import models

admin.site.register(models.AlbumModel)
admin.site.register(models.PhotoModel)