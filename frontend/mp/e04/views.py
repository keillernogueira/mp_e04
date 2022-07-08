from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse_lazy

from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib import messages

from django.http import JsonResponse
from django.core import serializers

from django.templatetags.static import static

from .forms import ProcessingForm, IdPersonForm, DetectionForm, UpdateDBForm, ConfigForm, FaceTrainForm, ObjectTrainForm

from .models import Database, Operation, OpConfig, GeneralConfig, Model, ImageDB, Processed, Output, Ranking, FullProcessed
from .filters import OperationFilter

import os
import shutil
import sys
import traceback
import inspect
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import cv2 as cv
from pathlib import Path
from zipfile import ZipFile
from fpdf import FPDF

from datetime import datetime
from .models import Operation
from .filters import OperationFilter
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from . import filters

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

from django.contrib.auth.decorators import login_required
from pessoas.manipulate_dataset import manipulate_dataset
from pessoas.train import train as train_face_func
from pessoas.retrieval import retrieval as face_retrieval
from pessoas.plots import plot_top15_person_retrieval
from objetos.yolov5.utils.data import img_formats, vid_formats
from objetos.yolov5.utils.options import defaultOpt
from objetos.yolov5.detect_obj import retrieval as detect_object
from objetos.yolov5.train_light import train as train_obj_func
from objetos.yolov5.utils.options import defaultOptTrain

from pessoas.retrieval import img_formats, vid_formats


# TODO tem uma flag debug do proprio django no seetings.py
debug = False


def extract_files_from_zip(zip_file, extract_path=Path('/tmp')):
    with ZipFile(zip_file, 'r') as zipObj:
        file_objects = [item for item in zipObj.namelist()
                        if os.path.splitext(item)[1].replace('.', '') in img_formats + vid_formats]

        for item in file_objects:
            zipObj.extract(item, path=extract_path)


def get_image_folder(request, form_data, operation, config):
    zip_file = request.FILES.get('zipFile', '')
    if zip_file != '':
        extractPath = Path(os.path.join(config.save_path, str(operation.id), 'input_files'))
        extractPath.mkdir(parents=True, exist_ok=True)
        extract_files_from_zip(zip_file, extractPath)
        img_folder = extractPath
    else:
        img_folder = Path(form_data['folderInput'])

    return str(img_folder)


def new_operation(request, form_data, optype="det"):
    op = {'det': Operation.OpType.DETECTION, 'ret': Operation.OpType.RETRIEVAL}
    fkey = {'det': 'doFaceRetrieval', 'ret': 'doObjectDetection'}

    operation = Operation()
    operation.user = request.user
    operation.type = op[optype] if not form_data[fkey[optype]] else Operation.OpType.RET_AND_DET
    operation.status = Operation.OpStatus.PENDING
    operation.save()

    return operation


def load_database_features(databases):
    db_features = {'feature_mean': np.array([[0.0]]),
                   'len': 0,
                   'feature': [],
                   'normalized_feature': [],
                   'name': [],
                   'image': [],
                   'id': []}

    for db in databases:
        features = ImageDB.objects.filter(database=db)
        database = Database.objects.filter(id=db)[0]

        new_len = db_features['len'] + database.quantity
        # db_features['feature_mean'] = (db_features['feature_mean'] * db_features['len'] + database.feature_mean * database.quantity) / new_len
        
        db_ft_mean = np.array(eval(database.feature_mean))        
        db_features['feature_mean'] = (db_features['feature_mean'] * db_features['len'] +
                                       db_ft_mean * database.quantity) / new_len
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
    db_features['image'] = np.array(db_features['image'])
    db_features['name'] = np.array(db_features['name'])
    db_features['id'] = np.array(db_features['id'])
    db_features['normalized_feature'] = db_features['feature'] - (db_features['feature_mean'] - 1e-18)
    db_features['normalized_feature'] = normalize(db_features['normalized_feature'], norm='l2', axis=1)
    #[feat - db_features['feature_mean'] for feat in db_features['feature']]

    return db_features


def save_retrieval_results(operation, data, confidence):
    out_data = []
    rkg_data = []
    
    for i, img in enumerate(data):
        for key, face in img.items():
            if 'face' not in key:
                continue
            prc = Processed(operation=operation, path=img['path'], hash=img['hash'],
                            frame=face['frame_num'] if 'frame_num' in face else 0)
            prc.save()
    
            out_bb = Output(processed=prc, parameter=Output.ParameterOpt.BB, value=repr(face['box']))
            out_data.append(out_bb)

            if face['confidence most similar'] < confidence:
                continue

            # r ranking, k name, person [score, imgdb_id]
            for r, (k, person) in enumerate(sorted(face['top options'].items(), key=lambda x: x[1][0], reverse=True)):
                # save only rankings higher than threshold
                if person[0] < confidence:
                    break
                imgdb = ImageDB.objects.filter(id=person[1])[0]
                ranking = Ranking(processed=prc, imagedb=imgdb, position=r+1, value=person[0])
                rkg_data.append(ranking)

    # Processed.objects.bulk_create(prc_data)
    Ranking.objects.bulk_create(rkg_data)
    Output.objects.bulk_create(out_data)


def save_detection_results(operation, data):
    out_data = []
    prc_data = []
    for i, img in enumerate(data):
        prc = Processed(operation=operation, path=img['path'], hash=img['hash'], frame=img['frame'])
        # prc_data.append(prc)
        prc.save()
        # print(img)
        for obj_id in range(1, img['objects'] + 1):
            obj = img[f'object_{obj_id}']
            out_bb = Output(processed=prc, parameter=Output.ParameterOpt.BB, value=repr(obj['box']), obj=obj_id-1)
            out_score = Output(processed=prc, parameter=Output.ParameterOpt.SCORE, value=repr(obj['confidence']),
                               obj=obj_id-1)
            out_label = Output(processed=prc, parameter=Output.ParameterOpt.LABEL, value=repr(obj['class']),
                               obj=obj_id-1)
            out_data.append(out_bb)
            out_data.append(out_score)
            out_data.append(out_label)

    # Processed.objects.bulk_create(prc_data)
    Output.objects.bulk_create(out_data)


def retrieval_process(operation, config_data, img_folder, database, confid_threshold):
    ret_model = Model.objects.filter(id=config_data.ret_model_id)[0]
    preprocessing = dict(GeneralConfig.PreProcess.choices)[config_data.ret_pre_process].lower()
    conf_thres = float(confid_threshold) / 100.0
    if ret_model.model_path == "":
        ret_model.model_path = None

    # Operation Configs
    operation_config_db = OpConfig(op=operation, parameter=OpConfig.ParameterOpt.DB, value=repr(database))
    operation_config_db.save()

    operation_config_rt = OpConfig(op=operation, parameter=OpConfig.ParameterOpt.RET_CONF, value=conf_thres)
    operation_config_rt.save()

    # Retrieval
    try:
        operation.status = Operation.OpStatus.PROCESSING
        operation.save()

        db_features = load_database_features(database)

        try:
            # run face retrieval
            data = face_retrieval(img_folder,
                                  db_features,
                                  os.path.join(config_data.save_path, str(operation.id), 'results'),
                                  input_data='image', output_method='json',
                                  model_name=ret_model.name, model_path=ret_model.model_path,
                                  preprocessing_method=preprocessing)
            # Saving in sql
            save_retrieval_results(operation, data, conf_thres)

        except Exception:
            operation.status = Operation.OpStatus.ERROR
            traceback.print_exc()
    except:
        operation.status = Operation.OpStatus.ERROR
        traceback.print_exc()


def detection_process(operation, config_data, img_folder, confid_threshold):
    det_model = Model.objects.filter(id=config_data.det_model_id)[0]
    det_options = defaultOpt()
    det_options.conf_thres = float(confid_threshold) / 100.0

    # Operation Configs
    operation_config_dt = OpConfig(op=operation, parameter=OpConfig.ParameterOpt.DET_CONF,
                                   value=det_options.conf_thres)
    operation_config_dt.save()

    # TODO Class Filters
    # det_options.classes = [1, 2, ...]

    # Detection
    try:
        operation.status = Operation.OpStatus.PROCESSING
        operation.save()
        # run obejct detection
        data = detect_object(img_folder, det_model.model_path,
                             os.path.join(config_data.save_path, str(operation.id), 'results'),
                             'both', opt=det_options)

        # Saving in sql
        save_detection_results(operation, data)
    except Exception:
        operation.status = Operation.OpStatus.ERROR
        traceback.print_exc()


@login_required
def index(request):
    ch = GeneralConfig.PreProcess.choices
    print(ch, dict(ch), dict(ch)['MT'].lower())
    return render(request, 'e04/index.html')


@login_required
def id_person(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if is_ajax and request.method == 'POST':
        form = IdPersonForm(request.POST, request.FILES, auto_id='%s')
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")

        if form.is_valid():
            form_data = form.cleaned_data

            operation = new_operation(request, form_data, optype='ret')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            img_folder = get_image_folder(request, form_data, operation, config_data)

            retrieval_process(operation, config_data, img_folder,
                              form_data['databases'], form_data['retrievalThreshold'])

            # Detection
            if form_data['doObjectDetection']:
                detection_process(operation, config_data, img_folder, form_data['detectionThreshold'])

            # If in thread it will be different
            if operation.status != Operation.OpStatus.ERROR:
                operation.status = Operation.OpStatus.FINISHED

            # redirect to a new URL:
            # return HttpResponseRedirect(reverse_lazy('results'))
            return JsonResponse({'result_id': operation.id}, status=200)
        else:
            # return render(request, 'e04/id_person.html', {'form': form})
            return JsonResponse({"error": form.errors}, status=400)
    else:
        form = IdPersonForm(auto_id='%s')
        return render(request, 'e04/id_person.html', {'form': form})


@login_required
def update_db(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if is_ajax and request.method == 'POST':
        form = UpdateDBForm(request.POST)

        # if new database, then the name field is required
        required = request.POST['database'] == '0'
        form.fields['dbName'].required = required

        if form.is_valid():
            op = Operation(user=request.user,
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
                                    path=feats['image'][i], hash=feats['hashes'][i], bb=repr(feats['bbs'][i].tolist()),
                                    features=repr(feats['normalized_feature'][i].tolist()), label=feats['name'][i]))
            ImageDB.objects.bulk_create(data)

            op.status = Operation.OpStatus.FINISHED
            op.save()

            init_qnt = db.quantity
            db.quantity = db.quantity + len(feats['name'])
            db_ft_mean = np.array(eval(db.feature_mean))
            db_ft_mean_to_save = (db_ft_mean * init_qnt + feats['feature_mean'] * len(feats['name'])) / db.quantity
            db.feature_mean = repr(db_ft_mean_to_save.tolist())
            db.save()

            # return HttpResponseRedirect(reverse_lazy('results'))
            return JsonResponse({'result_id': op.id}, status=200)
        else:
            # return render(request, 'e04/update_db.html', {'form': form})
            return JsonResponse({"error": form.errors}, status=400)
    else:
        form = UpdateDBForm()
        return render(request, 'e04/update_db.html', {'form': form})


@login_required
def detect_obj(request):
    # if this is a POST request we need to process the form data
    print(request.user, request.user.id)
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    print(request.POST, request.method, request.FILES)

    if is_ajax and request.method == "POST":
        form = DetectionForm(request.POST, request.FILES, auto_id='%s')
        form.fields['databases'].required = 'doFaceRetrieval' in form.data.keys()
        
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")
 
        if form.is_valid():
            form_data = form.cleaned_data

            operation = new_operation(request, form_data, optype='det')

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            img_folder = get_image_folder(request, form_data, operation, config_data)

            detection_process(operation, config_data, img_folder, form_data['detectionThreshold'])
            
            # Retrieval
            if form_data['doFaceRetrieval']:
                retrieval_process(operation, config_data, img_folder,
                                  form_data['databases'], form_data['retrievalThreshold'])

            # If in thread it will be different
            if operation.status != Operation.OpStatus.ERROR:
                operation.status = Operation.OpStatus.FINISHED
            # TODO: If op ERROR do something

            # redirect to a new URL:
            # return HttpResponseRedirect(reverse_lazy('results'))
            return JsonResponse({'result_id': operation.id}, status=200)
        else:
            # return render(request, 'e04/detect_obj.html', {'form': form})
            return JsonResponse({"error": form.errors}, status=400)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = DetectionForm(auto_id='%s')
        return render(request, 'e04/detect_obj.html', {'form': form})


@login_required
def results(request):
    current_user = request.user
    if current_user.is_staff or current_user.is_superuser:
        filtered_results_list = filters.OperationFilter(
                      request.GET, 
                      queryset=Operation.objects.all()
                  ).qs
        results_list = Operation.objects.get_queryset().all()
    else:
        filtered_results_list = filters.OperationFilter(
                      request.GET, 
                      queryset=Operation.objects.get_queryset().all().filter(user_id=current_user.id)
                  ).qs
        results_list = Operation.objects.get_queryset().all().filter(user_id=current_user.id)

    myFilter = OperationFilter(request.GET, queryset=results_list)

    paginator = Paginator(filtered_results_list,25)
    page = request.GET.get('page')
    try:
        response = paginator.page(page)
    except PageNotAnInteger:
        response = paginator.page(1)
    except EmptyPage:
        response = paginator.page(paginator.num_pages)

    '''context = {'results_list': results_list, 'myFilter': myFilter, 'response' : response}'''

    return render(request, 'e04/results.html', {'response': response,'myFilter': myFilter } )


def prepare_data_for_export(operation_id, num_ranks_saved):
    export_face_dict = {'File': [], 'Hash': [], 'BoundBox': [], 'Frame': []}
    for i in range(num_ranks_saved):
        export_face_dict['Rank' + str(i + 1) + '_label'] = []
        export_face_dict['Rank' + str(i + 1) + '_score'] = []
    export_detec_dict = {'File': [], 'Hash': [], 'BoundBox': [], 'Frame': [], 'Score': [], 'Label': []}

    processeds_list = Processed.objects.filter(operation__id=operation_id)
    count_videos = 0
    count_images = 0
    for processed in processeds_list:
        if processed.path.split('.')[-1] in img_formats:
            count_images += 1
        elif processed.path.split('.')[-1] in vid_formats:
            count_videos += 1
        outputs = Output.objects.filter(processed=processed)
        ranking = Ranking.objects.filter(processed=processed).order_by('position')
        if ranking:  # there is a raking to process
            export_face_dict['File'].append(processed.path)
            export_face_dict['Hash'].append(processed.hash)
            export_face_dict['Frame'].append(processed.frame)
            if outputs:  # there is an output to process
                export_face_dict['BoundBox'].append(outputs.first().value)
            for i in range(0, num_ranks_saved):
                try:
                    export_face_dict['Rank' + str(i + 1) + '_label'].append(ranking[i].imagedb.label)
                    export_face_dict['Rank' + str(i + 1) + '_score'].append(ranking[i].value)
                except IndexError:
                    export_face_dict['Rank' + str(i + 1) + '_label'].append("N/A")
                    export_face_dict['Rank' + str(i + 1) + '_score'].append("N/A")
        if outputs:
            bbs = [out for out in outputs if out.parameter == Output.ParameterOpt.BB]
            bbs.sort(key=lambda x: x.obj)
            scs = [out for out in outputs if out.parameter == Output.ParameterOpt.SCORE]
            scs.sort(key=lambda x: x.obj)
            lbl = [out for out in outputs if out.parameter == Output.ParameterOpt.LABEL]
            lbl.sort(key=lambda x: x.obj)

            for lb, sc, bb in zip(lbl, scs, bbs):
                export_detec_dict['File'].append(processed.path)
                export_detec_dict['Hash'].append(processed.hash)
                export_detec_dict['Frame'].append(processed.frame)

                export_detec_dict['BoundBox'].append(eval(bb.value))
                export_detec_dict['Score'].append(eval(sc.value))
                export_detec_dict['Label'].append(lb.value.replace("'", ""))

    return export_face_dict, export_detec_dict, len(processeds_list), count_images, count_videos


def export_xls(user, export_face_dict, export_detec_dict, operation_id, save_path):
    init = pd.DataFrame([user.username, user.first_name + ' ' + user.last_name, user.email,
                         Operation.objects.filter(id=operation_id).first().date.replace(tzinfo=None)])
    init.index = ['Username', 'Nome', 'Email', 'Data da execução']
    df1 = pd.DataFrame(export_face_dict)
    df2 = pd.DataFrame(export_detec_dict)

    with pd.ExcelWriter(os.path.join(save_path, 'exported.xlsx')) as writer:
        init.to_excel(writer, sheet_name='info', header=False, index=True)
        df1.to_excel(writer, sheet_name='person_id')
        df2.to_excel(writer, sheet_name='obj_detect')

    return os.path.join(save_path, 'report.xlsx')


def export_pdf(user, export_face_dict, export_detec_dict, total, t_img, t_videos, operation_id,
               save_path, num_ranks_saved):
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_margins(25, 25, 25)

        def header(self):
            self.image(os.path.join(os.getcwd(), 'e04', 'static', 'mpmg_logo.png'), 79, 15, 50)

            self.set_font('Arial', 'B', 16)
            self.ln(20)
            self.cell(0, 0, 'Relatório Automático', 0, 0, 'C')
            self.ln(14)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)

            self.cell(0, 0, 'Sistema -- ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 0, 0, 'L')
            self.cell(0, 0, 'Page ' + str(self.page_no()) + '/' + str(self.alias_nb_pages()), 0, 0, 'R')

        def bold_part_text(self, text_part1, text_part2, width, height=3, ln=2, fill=False, rgb=[211, 227, 230]):
            if fill is True:
                self.set_fill_color(rgb[0], rgb[1], rgb[2])
            self.set_font('Arial', 'B', 11)
            self.cell(width, height, text_part1, 0, 0, fill=fill)
            self.set_font('Arial', '', 11)
            self.multi_cell(0, height, text_part2, 0, 1, fill=fill)
            self.ln(ln)

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.bold_part_text('Software: ', 'Sistema para reconhecimento de pessoas e detecção de objetos', width=20)
    pdf.bold_part_text('Versão: ', '1.0 - 01/07/2022', width=16, ln=8)

    pdf.bold_part_text('Código identificador da análise: ', str(operation_id), width=60)
    pdf.bold_part_text('Total de registros processados: ', str(total), width=60)
    pdf.bold_part_text('Total de imagens processadas: ', str(t_img), width=59)
    pdf.bold_part_text('Total de vídeos processados: ', str(t_videos), width=55)
    pdf.bold_part_text('Data e hora: ', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), width=23, ln=8)

    pdf.bold_part_text('Usuário: ', user.first_name + ' ' + user.last_name, width=18)
    pdf.bold_part_text('E-mail: ', user.email, width=15, ln=18)

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 0, 'Resultado da Análise', 0, 0, 'C')
    pdf.ln(14)

    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Os resultados a seguir são fruto de um modelo probabilístico, '
                         'com o objetivo de auxiliar na triagem de imagens e vídeos. Portanto, '
                         'podem ocorrer falsos positivos ou falsos negativos. Necessário executar a '
                         'verificação visual.', 0, 'J', False)
    pdf.ln(12)

    count_id = 0
    for i in range(len(export_face_dict['File'])):
        img_flag = False
        if export_face_dict['File'][i].split('.')[-1] in img_formats:
            img_flag = True
        pdf.bold_part_text('ID: ', str(count_id), width=7, height=5, fill=True, rgb=[183, 206, 172])
        pdf.bold_part_text('Arquivo: ', export_face_dict['File'][i], width=18, height=5)
        pdf.bold_part_text('Tipo: ', 'Imagem' if img_flag else 'Vídeo', width=11)
        pdf.bold_part_text('Hash: ', export_face_dict['Hash'][i], width=13)
        if not img_flag:
            pdf.bold_part_text('Timestamp: ', export_face_dict['Frame'][i], width=15)

        string_rank = ''
        string_confidence = ''
        for j in range(num_ranks_saved):
            if export_face_dict['Rank' + str(j + 1) + '_label'][i] == 'N/A':
                break
            string_rank += export_face_dict['Rank' + str(j + 1) + '_label'][i] + ', '
            string_confidence += str("%.2f" % round(export_face_dict['Rank' + str(j + 1) + '_score'][i]*100, 2)) + ', '
        pdf.bold_part_text('Ranking: ', string_rank[:-2], width=17)
        pdf.bold_part_text('Confiança: ', string_confidence[:-2], width=21, ln=8)
        count_id += 1

    for i in range(len(export_detec_dict['File'])):
        img_flag = False
        if export_detec_dict['File'][i].split('.')[-1] in img_formats:
            img_flag = True
        pdf.bold_part_text('ID: ', str(count_id), width=7, height=5, fill=True)
        pdf.bold_part_text('Arquivo: ', export_detec_dict['File'][i], width=18, height=5)
        pdf.bold_part_text('Tipo: ', 'Imagem' if img_flag else 'Vídeo', width=11)
        pdf.bold_part_text('Hash: ', export_detec_dict['Hash'][i], width=13)
        if not img_flag:
            pdf.bold_part_text('Timestamp: ', export_detec_dict['Frame'][i], width=15)
        pdf.bold_part_text('Rótulo: ', FullProcessed.Detection.label_to_superlabel[export_detec_dict['Label'][i]],
                           width=16)
        pdf.bold_part_text('Confiança: ', str("%.2f" % round(export_detec_dict['Score'][i]*100, 2)), width=23, ln=8)
        count_id += 1

    pdf.output(os.path.join(save_path, 'report.pdf'), 'F')
    return os.path.join(save_path, 'report.pdf')

@login_required
def update_train_detail(request, operation_id):
    image_list = ImageDB.objects.filter(operation__id=operation_id)
    operation = Operation.objects.get(id=operation_id)
    database_images = ImageDB.objects.filter(operation__id=operation_id)[0].database

    context = {'image_list':image_list,'operation':operation, 'database':database_images}

    return render(request,'e04/update_train_detail.html',context)

@login_required
def detailed_result(request, operation_id):
    if request.method == 'POST':
        num_ranks_saved = 3
        config_data = GeneralConfig.objects.all()
        config_data = config_data[0] if len(config_data) else GeneralConfig()

        face_dict, detec_dict, total, t_img, t_videos = prepare_data_for_export(operation_id, num_ranks_saved)

        if request.POST.get("xls"):
            file = export_xls(request.user, face_dict, detec_dict, operation_id, config_data.save_path)
        else:
            file = export_pdf(request.user, face_dict, detec_dict, total, t_img, t_videos,
                              operation_id, config_data.save_path, num_ranks_saved)

        with open(file, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file)
            return response
    else:
        img_sz = 70
        config_data = GeneralConfig.objects.filter(id=1)
        config_data = config_data[0] if len(config_data) else GeneralConfig()

        # Getting tmp folder to copy results to
        root = os.path.split(os.path.abspath(__file__))[0]
        tmp = Path(os.path.join(root, static('')[1:], 'tmp', str(operation_id), 'results'))
        tmp.mkdir(parents=True, exist_ok=True)

        op = Operation.objects.filter(id=operation_id)[0]
        
        operations = {Operation.OpType.RET_AND_DET: ["ret", "det"],
                      Operation.OpType.RETRIEVAL: ["ret"], Operation.OpType.DETECTION: ["det"]}
        
        processeds_list = Processed.objects.filter(operation__id=operation_id)
        unique = set([prc.path for prc in processeds_list])

        formated_processed_list = {}

        for i, img in enumerate(unique):
            # TODO isso nao funciona pra video - @Pedro
            opimg = cv.imread(img)
            formated_processed_list[img] = FullProcessed(f"{operation_id}_{i}", path=img, w=opimg.shape[1],
                                                         h=opimg.shape[0], operation=operation_id,
                                                         detection_result_path=os.path.join(config_data.save_path,
                                                                                            str(operation_id), 'results',
                                                                                            os.path.basename(img)),)

        for processed in processeds_list:
            fprc = formated_processed_list[processed.path]

            outputs = Output.objects.filter(processed=processed)

            # If detection
            if len(outputs) % 3 == 0: 
                bbs = [out for out in outputs if out.parameter == Output.ParameterOpt.BB]
                bbs.sort(key=lambda x: x.obj)
                scs = [out for out in outputs if out.parameter == Output.ParameterOpt.SCORE]
                scs.sort(key=lambda x: x.obj)
                lbl = [out for out in outputs if out.parameter == Output.ParameterOpt.LABEL]
                lbl.sort(key=lambda x: x.obj)

                for lb, sc, bb in zip(lbl, scs, bbs):
                    rel_bb = eval(bb.value)
                    rel_bb = [rel_bb[0]/fprc.w, rel_bb[1]/fprc.h, rel_bb[2]/fprc.w, rel_bb[3]/fprc.h]
                    fprc.detections.append(FullProcessed.Detection(lb.value.replace("'", ""), eval(sc.value), rel_bb))

            # If face retrieval
            elif len(outputs) == 1:
                bbx = eval(outputs[0].value)
                rel_bbx = [bbx[0]/fprc.w, bbx[1]/fprc.h, bbx[2]/fprc.w, bbx[3]/fprc.h]
                face_id = len(fprc.faces)
                fprc.faces.append(FullProcessed.Faces(face_id, rel_bbx))
                face = fprc.faces[-1]

                ranking = Ranking.objects.filter(processed=processed)
                for r in ranking:
                    face.rankings.append(FullProcessed.Faces.Ranking(r.position, r.value, r.imagedb))
                
                face.rankings.sort(key=lambda x: x.position)

                ranking_img_info = [(r.value, r.imgdb.label, r.imgdb.path) for r in face.rankings]

        context = {'op': operation_id, 'processeds_list': processeds_list,
                   'formated_processed_list': formated_processed_list, 'operations': operations[op.type],
                   'tmp': static(os.path.join('tmp', str(operation_id), 'results')),
                   'w': img_sz*16, 'h': img_sz*9}

        return render(request, 'e04/detailed_result_v2.html',context)


def requestImageDB(request):
    # request should be ajax and method should be POST.
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if is_ajax and request.method == "POST":
        img_id = request.POST.get('img_id', '')

        if img_id == '':
            return JsonResponse({"error": ""}, status=400)

        imagedb = ImageDB.objects.filter(id=img_id)[0]

        root = os.path.split(os.path.abspath(__file__))[0]
        tmp = Path(os.path.join(root, static('')[1:], 'tmp', 'images'))
        tmp.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(tmp/os.path.basename(imagedb.path)):
            shutil.copy(imagedb.path, tmp/os.path.basename(imagedb.path))

        instance = {'tmp' : static(os.path.join('tmp', 'images')),
                    'path': os.path.basename(imagedb.path)}

        # send to client side.
        return JsonResponse({"instance": instance}, status=200)

    # some error occured
    return JsonResponse({"error": ""}, status=400)


def requestImagePath(request):
    # request should be ajax and method should be POST.
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if is_ajax and request.method == "POST":
        img_path = request.POST.get('path', '')

        if img_path == '':
            return JsonResponse({"error": ""}, status=400)

        if not os.path.exists(img_path):
            return JsonResponse({"error": ""}, status=400)

        root = os.path.split(os.path.abspath(__file__))[0]
        tmp = Path(os.path.join(root, static('')[1:], 'tmp', 'images'))
        tmp.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(tmp/os.path.basename(img_path)):
            shutil.copy(img_path, tmp/os.path.basename(img_path))

        instance = {'tmp': static(os.path.join('tmp', 'images')),
                    'path': os.path.basename(img_path)}

        # send to client side.
        return JsonResponse({"instance": instance}, status=200)

    # some error occured
    return JsonResponse({"error": ""}, status=400)


@login_required
def config(request):
    if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')

    try:
        data = GeneralConfig.objects.all()[0]
        # Get Values from database and load as initial form value
        form = ConfigForm(initial={'ret_pre_process': data.ret_pre_process, 'ret_model': data.ret_model,
                                'det_model': data.det_model, 'save_path': data.save_path})
    except:
        form = ConfigForm()

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
            messages.error(request, 'Falha ao salvar as configurações.')
            form = ConfigForm(initial={'ret_pre_process': data.ret_pre_process, 'ret_model': data.ret_model,
                                       'det_model': data.det_model, 'save_path': data.save_path})
    
    context = {'form': form}
    return render(request, 'e04/config.html', context)


@login_required
def train(request):
    if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')
    return HttpResponseRedirect('/e04/train/face')
    
@login_required
def train_face(request):
    if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')
    form = FaceTrainForm()

    if request.method == 'POST':
        form = FaceTrainForm(request.POST)
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")
        if form.is_valid():
            data = form.cleaned_data
            print(data)
            #dataset_path, save_dir, model_name, preprocessing_method='sphereface', resume_path=None, num_epoch=71
            model_name = 'curricularface'
            save_dir = ''
            dataset_path = ''
            num_epoch = data['num_epoch']
            op = Operation(user=request.user,
                           type=Operation.OpType.TRAIN,
                           status=Operation.OpStatus.PROCESSING)
            op.save()

            operation_config = OpConfig(op=op, parameter=OpConfig.ParameterOpt.TRAIN_EPOCH,
                                               value=num_epoch)
            operation_config.save()

            try:
                preprocessing_method = GeneralConfig.objects.all()[0].ret_pre_process
                if preprocessing_method == 'MT':
                    preprocessing_method = 'mtcnn'
                elif preprocessing_method == 'OP':
                    preprocessing_method = 'openface'
                elif preprocessing_method == 'SP':
                    preprocessing_method = 'sphereface'
                elif preprocessing_method == 'DE':
                    preprocessing_method = None
            except:
                preprocessing_method = 'sphereface'

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            dataset_path = get_image_folder(request, data, op, config_data)
            print('------------')
            print(dataset_path)

            if data['new_model']:
                resume_path = None
                save_dir = ''
                print('new')
                path = os.path.join(sys.path[0], 'pessoas', 'train_model')
                if not os.path.isdir(path):
                    os.mkdir(path)
                path = os.path.join(path, data['model_name'])
                if not os.path.isdir(path):
                    os.mkdir(path)
                save_dir = path
                print(save_dir)
                
            else:
                save_dir = os.path.join(sys.path[0], 'pessoas', 'train_model', data['model_sel'])
                resume_path = Model.objects.all().filter(name=data['model_sel'])[0].model_path

            try:
                print("Aaa")
                print(request.user)
                print(train_face.__code__.co_varnames)
                print(preprocessing_method)
                val_acc, save_name = train_face_func(dataset_path, save_dir, model_name, preprocessing_method, resume_path, num_epoch)
                print('a')

                m = Model(op_id = op, type=Model.ModelType.FACE, name=data['model_name'], model_path= save_name, val_acc = val_acc)
                m.save()
                print("model created with mAP =", m.val_acc)
            except Exception:
                    op.status = Operation.OpStatus.ERROR
                    op.save()
                    traceback.print_exc()

            # If in thread it will be different
            if op.status != Operation.OpStatus.ERROR:
                op.status = Operation.OpStatus.FINISHED
                op.save()
            else:
                messages.error(request, 'Falha no treino.')
                return HttpResponseRedirect("/e04/train/face")

            messages.success(request, 'Treino Iniciado com Sucesso')
            return HttpResponseRedirect("/e04/train/face")
        else:
            print('invalid')
            print(form.errors.as_data())
            err_msg = ''
            for key in form.errors.as_data():
                for msg in form.errors.as_data()[key]:
                    err_msg += msg.message + ' | '
            messages.error(request, err_msg)


    context = {'form': form}
    return render(request, 'e04/train_face.html', context)


@login_required
def train_object(request):
    if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')
    form = ObjectTrainForm()

    if request.method == 'POST':
        form = ObjectTrainForm(request.POST)
        if (request.POST.get('folderInput', '') == '') and (request.FILES.get('zipFile', '') == ''):
            form.add_error(None, "Either a zip file or a local folder should be informed.")
            form.add_error('folderInput', "*")
            form.add_error('zipFile', "*")
        if form.is_valid():
            data = form.cleaned_data
            print(data)
            #dataset_path, save_dir, model_name, preprocessing_method='sphereface', resume_path=None, num_epoch=71
            model_name = 'curricularface'
            save_dir = ''
            dataset_path = ''
            num_epoch = data['num_epoch']
            op = Operation(user=request.user,
                           type=Operation.OpType.TRAIN,
                           status=Operation.OpStatus.PROCESSING)
            op.save()

            operation_config = OpConfig(op=op, parameter=OpConfig.ParameterOpt.TRAIN_EPOCH,
                                               value=num_epoch)
            operation_config.save()

            config_data = GeneralConfig.objects.all()
            config_data = config_data[0] if len(config_data) else GeneralConfig()

            dataset_path = get_image_folder(request, data, op, config_data)
            print('------------')
            print(dataset_path)
            model_name = ''

            if data['new_model']:
                resume_path = None
                save_dir = ''
                print('new')
                path = os.path.join(sys.path[0], 'objetos', 'train_model')
                if not os.path.isdir(path):
                    os.mkdir(path)
                path = os.path.join(path, data['model_name'])
                model_name = data['model_name']
                if not os.path.isdir(path):
                    os.mkdir(path)
                save_dir = path
                print(save_dir)
                
            else:
                resume_path = Model.objects.all().filter(name=data['model_sel'])[0].model_path
                path = os.path.join(sys.path[0], 'objetos', 'train_model', data['model_sel'])
                model_name = data['model_sel']
                save_dir = path

            try:
                opt = defaultOptTrain()
                opt.epochs = num_epoch
                if not data['new_model']:
                  opt.weights = resume_path + '/weights/best.pt'
                print(save_dir)
                val_acc = train_obj_func(opt = opt, hyp_path = os.path.join(sys.path[0], 'objetos', 'yolov5', 'hyp.scratch.yaml'), 
                                        data=dataset_path, output_path=save_dir)
                val_acc = val_acc[2]
                
                model_dir = ''
                maxn = 0
                for f in os.listdir(os.path.join(sys.path[0], 'objetos', 'train_model')):
                  if f.startswith(model_name):
                    print(f)
                    maxn += 1
                model_dir = path + str(maxn)
                print(model_dir)
            
                

                m = Model(op_id = op, type=Model.ModelType.OBJECT, name=data['model_name'], model_path=model_dir, val_acc = val_acc)
                m.save()
                print("model created with mAP =", m.val_acc)
            except Exception:
                    op.status = Operation.OpStatus.ERROR
                    op.save()
                    traceback.print_exc()

            # If in thread it will be different
            if op.status != Operation.OpStatus.ERROR:
                op.status = Operation.OpStatus.FINISHED
                op.save()
            else:
                messages.error(request, 'Falha no treino.')
                return HttpResponseRedirect("/e04/train/object")

            messages.success(request, 'Treino Concluído com Sucesso')
            return HttpResponseRedirect("/e04/train/object")
        else:
            print('invalid')
            print(form.errors.as_data())
            err_msg = ''
            for key in form.errors.as_data():
                for msg in form.errors.as_data()[key]:
                    err_msg += msg.message + ' | '
            messages.error(request, err_msg)


    context = {'form': form}
    return render(request, 'e04/train_object.html', context)

@login_required
def login(request):
    return render(request, 'e04/login.html')
