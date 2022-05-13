from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    return render(request, 'e04/index.html')


def id_person(request):
    return render(request, 'e04/id_person.html')


def update_db(request):
    return render(request, 'e04/update_db.html')


def detect_obj(request):
    return render(request, 'e04/detect_obj.html')


def results(request):
    return render(request, 'e04/results.html')


def config(request):
    return render(request, 'e04/config.html')


def train(request):
    return render(request, 'e04/train.html')
