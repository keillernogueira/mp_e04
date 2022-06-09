from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse_lazy

from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib import messages

from .forms import ProcessingForm, IdPersonForm, DetectionForm, UpdateDBForm, ConfigForm
from .models import Database, Operation, GeneralConfig, Model

import os
import sys
import inspect
from pathlib import Path
from zipfile import ZipFile

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

from pessoas.manipulate_dataset import manipulate_dataset
from objetos.yolov5.utils.data import img_formats, vid_formats
from objetos.yolov5.utils.options import defaultOpt

def extractFilesFromZip(zip_file, extract_path=Path('/tmp')):
    with ZipFile(zip_file, 'r') as zipObj:
        file_objects = [item for item in zipObj.namelist() if os.path.splitext(item)[1].replace('.', '') in img_formats + vid_formats]

        for item in file_objects:
            zipObj.extract(item, path=extract_path)

def index(request):
    return render(request, 'e04/index.html')


def id_person(request):
    if request.method == 'POST':
        form = IdPersonForm(request.POST, request.FILES, auto_id='%s')
        print(form.data)
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")

        if form.is_valid():
            form_data = form.cleaned_data
            print(form_data)

            operation = Operation()
            operation.user = request.user
            operation.type = Operation.OpType.RETRIEVAL if not form_data['doObjectDetection'] else Operation.OpType.RET_AND_DET
            operation.status = Operation.OpStatus.PENDING
            operation.save()

            img_folder = Path('.')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            zip_file = request.FILES.get('zipFile', '')
            if zip_file != '':
                extractPath = Path(os.path.join(config_data.save_path, str(operation.id), 'input_files'))
                extractFilesFromZip(zip_file, extractPath)
                img_folder = extractPath
            else:
                img_folder = Path(form_data['folderInput'])

            ret_model = Model.objects.filter(id=config_data.ret_model_id)[0]
            print(ret_model.model_path, ret_model.name)
            ret_options = defaultOpt()
            ret_options.conf_thres = float(form_data['retrievalThreshold'])/100.0

            # Retrieval
            try:
                operation.status = Operation.OpStatus.PROCESSING
                operation.save()

                # Features in DB? Load features here
                # Need to change retrieval implementation
                
            except:
                operation.status = Operation.OpStatus.ERROR
            
            # Detection
            if form_data['doObjectDetection']:
                det_model = Model.objects.filter(id=config_data.det_model_id)[0]
                print(det_model.model_path)
                det_options = defaultOpt()
                det_options.conf_thres = float(form_data['detectionThreshold'])/100.0

                try:
                    # rodar o modelo de detecção
                    pass
                except:
                    operation.status = Operation.OpStatus.ERROR

            # If in thread it will be different
            if operation.status != Operation.OpStatus.ERROR:
                operation.status = Operation.OpStatus.FINISHED

            # redirect to a new URL:
            return HttpResponseRedirect(reverse_lazy('results'))
        else:
            return render(request, 'e04/id_person.html', {'form': form})
    else:
        form = IdPersonForm(auto_id='%s')
        return render(request, 'e04/id_person.html', {'form': form})


def update_db(request):
    if request.method == 'POST':
        form = UpdateDBForm(request.POST)

        # if new database, then the name field is required
        required = request.POST['database'] == '0'
        form.fields['dbName'].required = required

        if form.is_valid():
            print(form.cleaned_data)

            # # mudar isso para usar usuario que fez a requisicao
            # op = Operation(user=User.objects.get(id=0), type=Operation.OpType.UPDATE,
            #                status=Operation.OpStatus.PROCESSING)
            # op.save()
            #
            # # new db
            # if form.cleaned_data['database'] == '0':
            #     db = Database(name=form.cleaned_data['dbName'])
            #     db.save()
            # else:
            #     db = Database.objects.get(id=int(form.cleaned_data['database']))

            feats = manipulate_dataset(form.cleaned_data['folderInput'])
            print(feats.keys())

            return HttpResponseRedirect(reverse_lazy('results'))
        else:
            return render(request, 'e04/update_db.html', {'form': form})
    else:
        form = UpdateDBForm()
        return render(request, 'e04/update_db.html', {'form': form})


def detect_obj(request):
    # if this is a POST request we need to process the form data
    print(request.user, request.user.id)
    if request.method == 'POST':
        form = DetectionForm(request.POST, request.FILES, auto_id='%s')

        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")
 
        if form.is_valid():
            form_data = form.cleaned_data
            print(form_data)

            operation = Operation()
            operation.user = request.user
            operation.type = Operation.OpType.DETECTION if not form_data['doFaceRetrieval'] else Operation.OpType.RET_AND_DET
            operation.status = Operation.OpStatus.PENDING
            operation.save()

            img_folder = Path('.')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            zip_file = request.FILES.get('zipFile', '')
            if zip_file != '':
                extractPath = Path(os.path.join(config_data.save_path, str(operation.id), 'input_files'))
                extractFilesFromZip(zip_file, extractPath)
                img_folder = extractPath
            else:
                img_folder = Path(form_data['folderInput'])

            det_model = Model.objects.filter(id=config_data.det_model_id)[0]
            print(det_model.model_path)
            det_options = defaultOpt()
            det_options.conf_thres = float(form_data['detectionThreshold'])/100.0

            # TODO Class Filters
            # det_options.classes = [1, 2, ...]

            # Detection
            try:
                operation.status = Operation.OpStatus.PROCESSING
                operation.save()
                # rodar o modelo de detecção
                
            except:
                operation.status = Operation.OpStatus.ERROR
            
            # Retrieval
            if form_data['doFaceRetrieval']:
                ret_model = Model.objects.filter(id=config_data.ret_model_id)[0]
                print(ret_model.model_path, ret_model.name)

                # Features in DB? Load features here
                # Need to change retrieval implementation

                try:
                    # rodar o modelo de retrieval
                    pass
                except:
                    operation.status = Operation.OpStatus.ERROR

            # If in thread it will be different
            if operation.status != Operation.OpStatus.ERROR:
                operation.status = Operation.OpStatus.FINISHED

            # redirect to a new URL:
            return HttpResponseRedirect(reverse_lazy('results'))
        else:
            return render(request, 'e04/detect_obj.html', {'form': form})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = DetectionForm(auto_id='%s')
        return render(request, 'e04/detect_obj.html', {'form': form})


def results(request):
    return render(request, 'e04/results.html')


def config(request):
    '''if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')'''

    data = GeneralConfig.objects.all()[0]
    # Get Values from database and load as initial form value
    form = ConfigForm(initial={'ret_pre_process': data.ret_pre_process, 'ret_model': data.ret_model,
                               'det_model': data.det_model, 'save_path': data.save_path})
    print(data)

    if request.method == 'POST':
        print(request.POST)
        form = ConfigForm(request.POST)
        if form.is_valid():
            a = form.save(commit=False)
            print("eba")
            a.user = request.user
            # User = get_user_model()
            # user = User.objects.all()[0]
            # a.user = user
            a.save()
            print(request.user)
            messages.success(request, 'As configurações Foram Salvas com Sucesso.')
            return HttpResponseRedirect("/e04/config")
        else:
            print('invalid')
            print(form.errors)
            context = {'form': form, 'message': "Falha ao Salvar as Configurações.\n" + str(form.errors)}
            return render(request, 'e04/config.html', context)
    
    context = {'form': form}
    return render(request, 'e04/config.html', context)


def train(request):
    return render(request, 'e04/train.html')
