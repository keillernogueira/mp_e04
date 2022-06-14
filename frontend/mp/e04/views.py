from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse_lazy

from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib import messages

from .forms import ProcessingForm, IdPersonForm, DetectionForm, UpdateDBForm, ConfigForm
from .models import Database, Operation, GeneralConfig, Model, ImageDB, Processed, Output, Ranking

import os
import sys
import traceback
import inspect
import numpy as np
from pathlib import Path
from zipfile import ZipFile

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

from pessoas.manipulate_dataset import manipulate_dataset
from pessoas.retrieval import individual_retrieval as face_retrieval
from objetos.yolov5.utils.data import img_formats, vid_formats
from objetos.yolov5.utils.options import defaultOpt
from objetos.yolov5.detect_obj import retrieval as detect_object 


def extractFilesFromZip(zip_file, extract_path=Path('/tmp')):
    with ZipFile(zip_file, 'r') as zipObj:
        file_objects = [item for item in zipObj.namelist() if os.path.splitext(item)[1].replace('.', '') in img_formats + vid_formats]

        for item in file_objects:
            zipObj.extract(item, path=extract_path)

def getImageFolder(request, form_data, operation, config):
    img_folder = Path('.')
        
    zip_file = request.FILES.get('zipFile', '')
    if zip_file != '':
        extractPath = Path(os.path.join(config.save_path, str(operation.id), 'input_files'))
        extractFilesFromZip(zip_file, extractPath)
        img_folder = extractPath
    else:
        img_folder = Path(form_data['folderInput'])

    return img_folder

def newOperation(request, form_data, optype="det"):
    op = {'det': Operation.OpType.DETECTION, 'ret': Operation.OpType.RETRIEVAL}
    fkey = {'det': 'doFaceRetrieval', 'ret': 'doObjectDetection'}

    operation = Operation()
    operation.user = request.user
    operation.type = op[optype] if not form_data[fkey[optype]] else Operation.OpType.RET_AND_DET
    operation.status = Operation.OpStatus.PENDING
    operation.save()

    return operation
    
def loadDatabaseFeatures(databases):
    db_features = {}
    db_features['feature_mean'] = np.array([[0.0]])
    db_features['len'] = 0
    db_features['feature'] = []
    db_features['normalized_feature'] = []
    db_features['name'] = []
    db_features['image'] = []
    db_features['id'] = []
    for db in databases:
        
        features = ImageDB.objects.filter(database=db)
        database = Database.objects.filter(id=db)[0]
        # print(db, database.quantity, len(features))

        new_len = db_features['len'] + database.quantity
        # db_features['feature_mean'] = (db_features['feature_mean'] * db_features['len'] + database.feature_mean * database.quantity) / new_len
        
        db_ft_mean = np.array(eval(database.feature_mean))        
        db_features['feature_mean'] = (db_features['feature_mean'] * db_features['len'] + db_ft_mean * database.quantity) / new_len
        db_features['len'] = new_len

        img_features = [np.array(eval(feat.features)) for feat in features]
        img_path = [feat.path for feat in features]
        img_name = [feat.label for feat in features]
        img_id = [feat.id for feat in features]

        db_features['feature'] += img_features
        db_features['image'] += img_path
        db_features['name'] += img_name
        db_features['id'] += img_id

    db_features['feature'] = np.array(db_features['feature'])
    db_features['normalized_feature'] = db_features['feature'] - (db_features['feature_mean'] - 1e-18)
    #[feat - db_features['feature_mean'] for feat in db_features['feature']]

    return db_features

def saveRetrievalResults(operation, data, confidence):
    out_data = []
    rkg_data = []
    prc_data = []
    for i, img in enumerate(data):
        for key, face in data.items():
            if 'face' not in key: continue
            if face['confidence most similar'] < confidence: continue
            prc = Processed(operation=operation, path=img['path'])
            prc_data.append(prc)
            
            out_bb = Output(processed=prc, parameter=Output.ParameterOpt.BB, value=repr(face['box']))
            out_data.append(out_bb)

            # r ranking, k name, person [score, imgdb_id]
            for r, (k, person) in enumerate(sorted(face['top options'].items(), key=lambda x: x[1][0], reverse=True)):
                # save only rankings higher than threshold
                if person[0] < confidence: break
                # imgdb = ImageDB.objects.filter(id=person[1])[0]
                ranking = Ranking(processed=prc, imagedb=person[1], position=r+1, value=person[0])
                rkg_data.append(ranking)

    Processed.objects.bulk_create(prc_data)
    Ranking.objects.bulk_create(rkg_data)
    Output.objects.bulk_create(out_data)

def saveDetectionResults(operation, data):
    out_data = []
    prc_data = []
    for i, img in enumerate(data):
        prc = Processed(operation=operation, path=img['path'], frame=img['frame'])
        prc_data.append(prc)
        for obj_id in range(1, data['objects'] + 1):
            obj = data[f'object_{obj_id}']
            out_bb = Output(processed=prc, parameter=Output.ParameterOpt.BB, value=repr(obj['box']))
            out_score = Output(processed=prc, parameter=Output.ParameterOpt.SCORE, value=repr(obj['confidence']))
            out_label = Output(processed=prc, parameter=Output.ParameterOpt.LABEL, value=repr(obj['class']))
            out_data.append(out_bb)
            out_data.append(out_score)
            out_data.append(out_label)

    Processed.objects.bulk_create(prc_data)
    Output.objects.bulk_create(out_data)


def index(request):
    ch = GeneralConfig.PreProcess.choices
    print(ch, dict(ch), dict(ch)['MT'].lower())
    return render(request, 'e04/index.html')

debug = False
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

            operation = newOperation(request, form_data, optype='ret')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            img_folder = getImageFolder(request, form_data, operation, config_data)

            ret_model = Model.objects.filter(id=config_data.ret_model_id)[0]
            preprocessing = dict(GeneralConfig.PreProcess.choices)[config_data.ret_pre_process].lower()
            conf_thres = float(form_data['retrievalThreshold'])/100.0
            print(ret_model.model_path, ret_model.name)

            # Retrieval
            try:
                operation.status = Operation.OpStatus.PROCESSING
                operation.save()
                
                db_features = loadDatabaseFeatures(form_data['databases'])

                if not debug:
                    try:
                        # run face retrieval
                        data = face_retrieval(img_folder,
                                              db_features,
                                              os.path.join(config_data.save_path, str(operation.id), 'results'),
                                              input_data='image', output_method='json', 
                                              model_name=ret_model.name, model_path=ret_model.model_path,
                                              preprocessing_method=preprocessing)
                        # Saving in sql
                        saveRetrievalResults(operation, data, conf_thres)

                    except Exception:
                        operation.status = Operation.OpStatus.ERROR
                        traceback.print_exc()
                
            except:
                operation.status = Operation.OpStatus.ERROR
            
            # Detection
            if form_data['doObjectDetection']:
                det_model = Model.objects.filter(id=config_data.det_model_id)[0]
                det_options = defaultOpt()
                det_options.conf_thres = float(form_data['detectionThreshold'])/100.0
                print(det_model.model_path)

                if not debug:
                    try:
                        operation.status = Operation.OpStatus.PROCESSING
                        operation.save()
                        
                        # run obejct detection 
                        data = detect_object(img_folder,
                                                    det_model.model_path,
                                                    os.path.join(config_data.save_path, str(operation.id), 'results'),
                                                    'both', opt=det_options)
                        # Saving in sql
                        saveDetectionResults(operation, data)
                    except Exception:
                        operation.status = Operation.OpStatus.ERROR
                        traceback.print_exc()

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
            # mudar isso para usar usuario que fez a requisicao
            op = Operation(user=request.user, #User.objects.get(id=1)
                           type=Operation.OpType.UPDATE,
                           status=Operation.OpStatus.PROCESSING)
            op.save()

            # new db
            if form.cleaned_data['database'] == '0':
                db = Database(name=form.cleaned_data['dbName'])
                db.save()
            else:
                db = Database.objects.get(id=int(form.cleaned_data['database']))

            feats = manipulate_dataset(form.cleaned_data['folderInput'])
            print(feats.keys())

            data = []
            for i in range(len(feats['name'])):
                data.append(ImageDB(operation=op, database=db,
                                    path=feats['image'][i], bb=repr(feats['bbs'][i].tolist()),
                                    features=repr(feats['feature'][i].tolist()), label=feats['name'][i]))
            ImageDB.objects.bulk_create(data)

            op.status = Operation.OpStatus.FINISHED
            op.save()

            init_qnt = db.quantity
            db.quantity = db.quantity + len(feats['name'])
            db_ft_mean = np.array(eval(db.feature_mean))
            db_ft_mean_to_save = (db_ft_mean * init_qnt + feats['feature_mean'] * len(feats['name'])) / db.quantity
            db.feature_mean = repr(db_ft_mean_to_save.tolist())
            db.save()

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
        form.fields['databases'].required = 'doFaceRetrieval' in form.data.keys()
        
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")
 
        if form.is_valid():
            form_data = form.cleaned_data
            print(form_data)

            operation = newOperation(request, form_data, optype='det')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            img_folder = getImageFolder(request, form_data, operation, config_data)
            print(img_folder)

            det_model = Model.objects.filter(id=config_data.det_model_id)[0]
            det_options = defaultOpt()
            det_options.conf_thres = float(form_data['detectionThreshold'])/100.0
            print(det_model.model_path)

            # TODO Class Filters
            # det_options.classes = [1, 2, ...]

            # Detection
            try:
                operation.status = Operation.OpStatus.PROCESSING
                operation.save()
                # run obejct detection 
                if not debug:
                    data = detect_object(img_folder,
                                                det_model.model_path,
                                                os.path.join(config_data.save_path, str(operation.id), 'results'),
                                                'both', opt=det_options)
                    # Saving in sql
                    saveDetectionResults(operation, data)
            except Exception:
                operation.status = Operation.OpStatus.ERROR
                traceback.print_exc()
            
            # Retrieval
            if form_data['doFaceRetrieval']:
                ret_model = Model.objects.filter(id=config_data.ret_model_id)[0]
                preprocessing = dict(GeneralConfig.PreProcess.choices)[config_data.ret_pre_process].lower()
                conf_thres = float(form_data['retrievalThreshold'])/100.0
                print(ret_model.model_path, ret_model.name, preprocessing)

                # Features in DB? Load features here
                db_features = loadDatabaseFeatures(form_data['databases'])
                
                # for k, v in db_features.items():
                #     print(k, len(v) if type(v) is list else v)

                if not debug:
                    try:
                        # run face retrieval
                        data = face_retrieval(img_folder,
                                              db_features,
                                              os.path.join(config_data.save_path, str(operation.id), 'results'),
                                              input_data='image', output_method='json', 
                                              model_name=ret_model.name, model_path=ret_model.model_path,
                                              preprocessing_method=preprocessing)
                        # Saving in sql
                        saveRetrievalResults(operation, data, conf_thres)

                    except Exception:
                        operation.status = Operation.OpStatus.ERROR
                        traceback.print_exc()

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
