from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from django.urls import reverse_lazy

from .forms import ProcessingForm, IdPersonForm, DetectionForm, UpdateDBForm
from .models import Database

from .forms import ConfigForm
from .models import GeneralConfig
from django.contrib.auth import get_user_model
from django.contrib import messages



def index(request):
    return render(request, 'e04/index.html')


def id_person(request):
    if request.method == 'POST':
        pass
    else:
        databases = Database.objects.all()
        return render(request, 'e04/id_person.html', {'databases': databases})


def update_db(request):
    if request.method == 'POST':
        form = UpdateDBForm(request.POST)
        # print(form)
        # print(form['database'])
        # print(form['folderInput'])
        print('---', form.is_valid())
        if form.is_valid():
            print(form)
            print('----------------------------------------------')
            print(form.cleaned_data)
            return HttpResponseRedirect(reverse_lazy('results'))
        else:
            return render(request, 'e04/update_db.html', {'form': form})
    else:
        form = UpdateDBForm()
        return render(request, 'e04/update_db.html', {'form': form})


def detect_obj(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        form = ProcessingForm(request.POST, request.FILES)
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('e04/results.html')

    # if a GET (or any other method) we'll create a blank form
    else:
        pr_form = ProcessingForm()
        det_form = DetectionForm()

    return render(request, 'e04/detect_obj.html', {'pr_form': pr_form, 'det_form': det_form})

    # return render(request, 'e04/detect_obj.html')


def results(request):
    return render(request, 'e04/results.html')


def config(request):

    '''if not request.user.is_superuser:
        return render(request, 'e04/permissiondenied.html')'''

    data=GeneralConfig.objects.all()[0]
    # Get Values from database and load as initial form value
    form = ConfigForm(initial = {'ret_pre_process': data.ret_pre_process, 'ret_model': data.ret_model, 'det_model': data.det_model, 'save_path': data.save_path})
    print(data)



    if request.method == 'POST':
        print("Scooby")
        print(request.POST)
        form = ConfigForm(request.POST)
        if form.is_valid():
            a = form.save(commit=False)
            print("eba")
            a.user = request.user
            #User = get_user_model()
            #user = User.objects.all()[0]
            #a.user = user
            a.save()
            print(request.user)
            messages.success(request, 'As configurações Foram Salvas com Sucesso.')
            return HttpResponseRedirect("/e04/config")
        else:
            print('invalid')
            print(form.errors)
            context = {'form': form,
                        'message': "Falha ao Salvar as Configurações.\n" + str(form.errors)}
            return render(request, 'e04/config.html', context)

    
    context = {'form': form}
    print("Arri egua")
    return render(request, 'e04/config.html', context)


def train(request):
    return render(request, 'e04/train.html')
