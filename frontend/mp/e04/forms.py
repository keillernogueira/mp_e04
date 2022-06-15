import os
from pathlib import Path

from .models import Database
from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from django.forms import ModelForm
from .models import GeneralConfig


def validateFolder(value):
    if not os.path.exists(value):
        raise ValidationError(message=f'{value} folder is invalid or it doesn\'t have permission to access.')

def dbs_as_choices(insert_new=True):
    choices = []
    if insert_new: 
        choices.append(['', ''])
        choices.append([0, 'Novo Banco'])
    for db in Database.objects.all():
        choices.append([db.pk, db.name])
    return choices

class ProcessingForm(forms.Form):
    zipFile = forms.FileField(required=False,
                              widget=forms.FileInput(attrs={'class': 'form-control', 
                                                            'onchange': 'toggleFolder(this)', 'accept': '.zip, .arj, .rar, .tar.gz, .tgz',
                                                            'required': True}))
    folderInput = forms.CharField(label='Pasta', validators=[validateFolder], required=False,
                                  widget=forms.TextInput(attrs={'class': 'form-control input-lg', 'placeholder': '/home', 
                                                               'onchange': 'toggleZip(this)',
                                                               'required': True}))


class DetectionForm(ProcessingForm):
    detectionThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=25,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))
    doFaceRetrieval = forms.BooleanField(label=u'Realizar reconhecimento de pessoas?', required=False,
                                            widget=forms.CheckboxInput(attrs={'class': 'form-check-input',
                                                                             'onclick': 'toggleRet()'}))

    # Retrieval configs
    databases = forms.MultipleChoiceField(label='Banco de dados onde procurar:', required=True,
                                 choices=dbs_as_choices(insert_new=False),
                                 widget=forms.SelectMultiple(attrs={'class': 'form-select custom-select', 'style': 'display: none;'}))
    retrievalThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=50,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))

class IdPersonForm(ProcessingForm):
    databases = forms.MultipleChoiceField(label='Banco de dados onde procurar:', required=True,
                                 choices=dbs_as_choices(insert_new=False),
                                 widget=forms.SelectMultiple(attrs={'class': 'form-select custom-select', 'style': 'display: none;'}))
    retrievalThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=50,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))

    doObjectDetection = forms.BooleanField(label=u'Realizar detecção de objetos?', required=False,
                                            widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 
                                                                              'onclick': 'toggleDet()',}))

    # Detection configs
    detectionThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=25,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))


class UpdateDBForm(forms.Form):
    # database = forms.ModelChoiceField(label='Banco de dados a ser atualizado:',
    #                                   queryset=Database.objects.all(), required=True,
    #                                   empty_label="Novo Banco",
    #                                   widget=forms.Select(attrs={'class': 'form-select'}))

    database = forms.ChoiceField(label='Banco de dados a ser atualizado:', required=True,
                                 choices=dbs_as_choices(),
                                 widget=forms.Select(attrs={'class': 'form-select', 'onchange': "newDB();"}))
    dbName = forms.CharField(label='Nome do novo banco:', required=False, min_length=1,
                             widget=forms.TextInput(attrs={'class': 'form-control input-lg'}))

    folderInput = forms.CharField(label='Dado a ser processado:', validators=[validateFolder], required=True,
                                  widget=forms.TextInput(attrs={'class': 'form-control input-lg', 'placeholder': '/home'}))


class ConfigForm(ModelForm):
    class Meta:
        model = GeneralConfig
        fields = ('save_path', "ret_model", 'ret_pre_process', "det_model")

        widgets = {
            'save_path': forms.TextInput(attrs={'class': 'form-control'}),
            'ret_model': forms.Select(attrs={'class': 'select-form'}),
            'det_model': forms.Select(attrs={'class': 'select-form'}),
            'ret_pre_process': forms.Select(attrs={'class': 'select-form'})
        }
