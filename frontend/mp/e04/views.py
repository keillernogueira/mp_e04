from django.http import HttpResponse
from django.shortcuts import render
from .forms import DetectionForm

def index(request):
    return render(request, 'e04/index.html')


def id_person(request):
    return render(request, 'e04/id_person.html')


def update_db(request):
    return render(request, 'e04/update_db.html')


def detect_obj(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        form = DetectionForm(request.POST, request.FILES)
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('e04/results.html')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = DetectionForm()

    return render(request, 'e04/detect_obj.html', {'form': form})

    # return render(request, 'e04/detect_obj.html')


def results(request):
    return render(request, 'e04/results.html')


def config(request):
    return render(request, 'e04/config.html')


def train(request):
    return render(request, 'e04/train.html')
