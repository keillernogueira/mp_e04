import os

from pathlib import Path

from .models import Database
from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _



def validateFolder(value):
    if not os.path.exists(value):
        raise ValidationError(message=f'{value} folder is invalid or doesn\'t have permission to access.')


class ProcessingForm(forms.Form):
    zipFile = forms.FileField()
    folderInput = forms.CharField(label='Pasta', validators=[validateFolder])


class DetectionForm(forms.Form):
    detectionThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=100, initial=50)


class IdPersonForm(forms.Form):
    database = forms.ChoiceField()
    retrievalThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=100, initial=50)


class UpdateDBForm(forms.Form):
    database = forms.ModelChoiceField(label='Banco de dados a ser atualizado:',
                                      queryset=Database.objects.all(), required=True,
                                      empty_label="Novo Banco",
                                      widget=forms.Select(attrs={'class': 'form-select'}))
    folderInput = forms.CharField(label='Dado a ser processado:', validators=[validateFolder], required=True,
                                  widget=forms.TextInput(attrs={'class': 'form-control input-lg'}))
