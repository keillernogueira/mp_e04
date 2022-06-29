from logging import PlaceHolder
import django_filters
from django_filters import DateFilter
from .models import Operation
from django_filters import DateFromToRangeFilter




class OperationFilter (django_filters.FilterSet):
    type_choices = (
        ('TR','Treino'),
        ('RE','Recuperação'),
        ('DE','Detecção'),
        ('UD','Update DB'),
        ('RD', 'Recuperação e Detecção')
    )
    teste_date = DateFromToRangeFilter (field_name='date', label ='Intervalo de datas', widget=django_filters.widgets.RangeWidget(attrs={'placeholder': 'dd/mm/aaaa'}))
    type = django_filters.ChoiceFilter (choices=type_choices, empty_label = 'Todos', label='Tipo')

    
    class Meta():
        model = Operation
        fields = ['id','type']
    


        
    

       
        
        

    
    

