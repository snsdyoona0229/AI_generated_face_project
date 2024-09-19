from django.db import models

class AlbumModel(models.Model):
    adate = models.DateTimeField(auto_now=True)
    alocation = models.CharField(max_length=200, blank=True, default='')
    atitle = models.CharField(max_length=100, null=False)
    adesc = models.TextField(blank=True, default='')
    def __str__(self):
        return self.atitle

class PhotoModel(models.Model):
    palbum = models.ForeignKey('AlbumModel', on_delete=models.CASCADE)
    psubject = models.CharField(max_length=100, null=False)
    pdate = models.DateTimeField(auto_now=True)
    purl = models.CharField(max_length=100, null=False)
    phit = models.IntegerField(default=0)
    def __str__(self):
        return self.psubject
