from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
import os

def validateFolder(value):
    if not os.path.exists(value):
        raise ValidationError(message=
            f'{value} folder is invalid or doesn\'t have permission to access.'
        )

class DetectionForm(forms.Form):
    zipFile = forms.FileField()
    folderInput = forms.CharField(label='Pasta', validators=[validateFolder])
    detectionThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=100, initial=50)