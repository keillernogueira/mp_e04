import django_filters
from django_filters import DateFilter
from matplotlib import widgets
from matplotlib.widgets import Widget
from .models import Operation
from django_filters import DateFromToRangeFilter
from django_filters import CharFilter
from django import forms

class OperationFilter (django_filters.FilterSet):
    type_choices = (
        ('TR','Treino'),
        ('RE','Recuperação'),
        ('DE','Detecção'),
        ('UD','Update DB'),
        ('RD', 'Recuperação e Detecção')
    )
    teste_date = DateFromToRangeFilter (field_name='date', label ='Intervalo de datas')
    type = django_filters.ChoiceFilter (choices=type_choices, empty_label = 'Todos', label='Tipo',)

    
    class Meta():
        model = Operation
        fields = ['id','type' ]
        
    

       
        
        

    
    