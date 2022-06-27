from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),

    path('id_person/', views.id_person, name='id_person'),
    path('update_db/', views.update_db, name='update_db'),

    path('detect_obj/', views.detect_obj, name='detect_obj'),

    # potencial para usar o list generic aqui
    path('results/', views.results, name='results'),
    path('results/<int:operation_id>/', views.detailed_result, name='detailed_result' ),
    path('results/imgdb', views.requestImageDB, name='detailed_result_imgdb' ),

    path('config/', views.config, name='config'),
    path('train/', views.train, name='train'),
]
