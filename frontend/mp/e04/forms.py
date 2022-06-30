import os
from pathlib import Path

from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from django.forms import ModelForm
from django.db import models

from .models import GeneralConfig, Database, Model


def validateFolder(value):
    if not os.path.exists(value):
        raise ValidationError(message=f'{value} folder is invalid or it doesn\'t have permission to access.')


def dbs_as_choices(insert_new=True):
    choices = []
    if insert_new:
        choices.append(['', ''])
        choices.append([0, 'Novo Banco'])
    for db in Database.objects.all():
        choices.append([db.id, db.name])
    return choices


class ProcessingForm(forms.Form):
    zipFile = forms.FileField(required=False,
                              widget=forms.FileInput(attrs={'class': 'form-control', 
                                                            'onchange': 'toggleFolder(this)',
                                                            'accept': '.zip, .arj, .rar, .tar.gz, .tgz',
                                                            'required': True}))
    folderInput = forms.CharField(label='Pasta', validators=[validateFolder], required=False,
                                  widget=forms.TextInput(attrs={'class': 'form-control input-lg',
                                                                'placeholder': '/home',
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
    retrievalThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=25,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['databases'].choices = dbs_as_choices(insert_new=False)


class IdPersonForm(ProcessingForm):
    databases = forms.MultipleChoiceField(label='Banco de dados onde procurar:', required=True,
                                 choices=dbs_as_choices(insert_new=False),
                                 widget=forms.SelectMultiple(attrs={'class': 'form-select custom-select', 'style': 'display: none;'}))
    retrievalThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=25,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))

    doObjectDetection = forms.BooleanField(label=u'Realizar detecção de objetos?', required=False,
                                           widget=forms.CheckboxInput(attrs={'class': 'form-check-input',
                                                                             'onclick': 'toggleDet()'}))
    # Detection configs
    detectionThreshold = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=25,
                                            widget=forms.NumberInput(attrs={'class': 'form-control', }))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['databases'].choices = dbs_as_choices(insert_new=False)


class UpdateDBForm(forms.Form):
    # database = forms.ModelChoiceField(label='Banco de dados a ser atualizado:',
    #                                   queryset=Database.objects.all(), required=True,
    #                                   empty_label="Novo Banco",
    #                                   widget=forms.Select(attrs={'class': 'form-select'}))

    database = forms.ChoiceField(label='Banco de dados a ser atualizado:', required=True,
                                 # choices=dbs_as_choices(),
                                 widget=forms.Select(attrs={'class': 'form-select', 'onchange': "newDB();"}))
    dbName = forms.CharField(label='Nome do novo banco:', required=False, min_length=1,
                             widget=forms.TextInput(attrs={'class': 'form-control input-lg'}))

    folderInput = forms.CharField(label='Dado a ser processado:', validators=[validateFolder], required=True,
                                  widget=forms.TextInput(attrs={'class': 'form-control input-lg',
                                                                'placeholder': '/home'}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['database'].choices = dbs_as_choices()


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


class FaceTrainForm(ProcessingForm):

    def __init__(self, *args, **kwargs):
        super(FaceTrainForm, self).__init__(*args, **kwargs)
        data = Model.objects.all().filter(type='FA')
        self.fields['model_sel'].choices = [(x.name, x.name) for x in data]
        self.fields['model_sel'].choices.insert(0, ('', 'Selecione um Modelo'))

    model_sel = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select'}), label='escolha', required=False)
    model_name = forms.CharField(label='Nome do novo banco:', required=False, min_length=1,
                             widget=forms.TextInput(attrs={'class': 'form-control input-lg',
                                                           'placeholder': 'Nomeie o Novo Modelo'}))
    new_model = forms.BooleanField(label=u'Criar Novo Modelo', required=False,
                                   widget=forms.CheckboxInput(attrs={'class': 'form-check-input',
                                                                     'onclick': 'toggleRet()'}))

    num_epoch = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=50,
                                   widget=forms.NumberInput(attrs={'class': 'form-control', }))

    def clean(self):
        cleaned_data = super(FaceTrainForm, self).clean()

        print(cleaned_data)

        if 'num_epoch' not in cleaned_data.keys() or not cleaned_data['num_epoch']:
            raise forms.ValidationError("Defina o número de epochs.")

        if cleaned_data['new_model'] and not cleaned_data['model_name']:
            raise forms.ValidationError("Insira um nome para o novo modelo.")

        if not cleaned_data['new_model'] and not cleaned_data['model_sel']:
            raise forms.ValidationError("Selecione o modelo a ser treinado.")

        if cleaned_data['new_model']:
            models = Model.objects.filter(name = cleaned_data['model_name'])
            if len(models) > 1:
                raise forms.ValidationError("Nome de modelo já existente.")

        if not cleaned_data['zipFile'] and 'folderInput'not in cleaned_data.keys():
            raise forms.ValidationError("Caminho de arquivo inválido.")

        elif cleaned_data['zipFile'] and cleaned_data['folderInput']:
            raise forms.ValidationError("Apenas um conjunto de imagens pode ser utilizado.")

        return cleaned_data

class ObjectTrainForm(ProcessingForm):

    def __init__(self, *args, **kwargs):
        super(ObjectTrainForm, self).__init__( *args, **kwargs)
        data = Model.objects.all().filter(type='OB')
        self.fields['model_sel'].choices = [(x.name, x.name) for x in data]
        self.fields['model_sel'].choices.insert(0, ('', 'Selecione um Modelo'))

    model_sel = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select'}), label='escolha', required=False)
    model_name = forms.CharField(label='Nome do novo banco:', required=False, min_length=1,
                             widget=forms.TextInput(attrs={'class': 'form-control input-lg',
                                                           'placeholder': 'Nomeie o Novo Modelo'}))
    new_model = forms.BooleanField(label=u'Criar Novo Modelo', required=False,
                                   widget=forms.CheckboxInput(attrs={'class': 'form-check-input',
                                                                     'onclick': 'toggleRet()'}))

    num_epoch = forms.IntegerField(label=u'Confiança mínima:', min_value=0, max_value=99, initial=50,
                                   widget=forms.NumberInput(attrs={'class': 'form-control', }))

    def clean(self):
        cleaned_data = super(FaceTrainForm, self).clean()

        print(cleaned_data)

        if 'num_epoch' not in cleaned_data.keys() or not cleaned_data['num_epoch']:
            raise forms.ValidationError("Defina o número de epochs.")

        if cleaned_data['new_model'] and not cleaned_data['model_name']:
            raise forms.ValidationError("Insira um nome para o novo modelo.")

        if not cleaned_data['new_model'] and not cleaned_data['model_sel']:
            raise forms.ValidationError("Selecione o modelo a ser treinado.")

        if cleaned_data['new_model']:
            models = Model.objects.filter(name = cleaned_data['model_name'])
            if len(models) > 1:
                raise forms.ValidationError("Nome de modelo já existente.")

        if not cleaned_data['zipFile'] and 'folderInput'not in cleaned_data.keys():
            print('aad')
            raise forms.ValidationError("Caminho de arquivo inválido.")

        elif cleaned_data['zipFile'] and cleaned_data['folderInput']:
            raise forms.ValidationError("Apenas um conjunto de imagens pode ser utilizado.")

        return cleaned_data

