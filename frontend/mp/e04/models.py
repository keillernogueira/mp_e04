from django.db import models
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _


class Operation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # https://stackoverflow.com/questions/54802616/how-to-use-enums-as-a-choice-field-in-django-model
    class OpType(models.TextChoices):
        TRAIN = 'TR', _('Treino')
        RETRIEVAL = 'RE', _('Recuperação')
        DETECTION = 'DE', _('Detecção')
        UPDATE = 'UD', _('Update DB')
        RET_AND_DET = 'RD', _('Recuperação e Detecção')
    type = models.CharField(max_length=2, choices=OpType.choices)

    class OpStatus(models.TextChoices):
        PROCESSING = 'PR', _('Processing')
        FINISHED = 'FI', _('Finished')
        PENDING = 'PD', _('Pending')
        ERROR = 'ER', _('Error')
    status = models.CharField(max_length=2, choices=OpStatus.choices)
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.type, self.status


class OpConfig(models.Model):
    op = models.ForeignKey(Operation, on_delete=models.CASCADE)

    class ParameterOpt(models.TextChoices):
        DB = 'DB', _('Database')
        RET_CONF = 'RC', _('Retrieval Confidence')
        DET_CONF = 'DC', _('Detection Confidence')
        TRAIN_EPOCH = 'EP', _('Train Epoch')
    parameter = models.CharField(max_length=2, choices=ParameterOpt.choices)

    value = models.CharField(max_length=100)


class Model(models.Model):
    # https://vertabelo.com/blog/one-to-one-relationship-in-database/
    op_id = models.OneToOneField(Operation, on_delete=models.CASCADE)

    class ModelType(models.TextChoices):
        FACE = 'FA', _('Face')
        OBJECT = 'OB', _('Object')
    type = models.CharField(max_length=2, choices=ModelType.choices)

    name = models.CharField(max_length=100)
    model_path = models.CharField(max_length=255)
    val_acc = models.FloatField()


class GeneralConfig(models.Model):
    ret_model = models.ForeignKey(Model, related_name='ret_model', on_delete=models.CASCADE)
    det_model = models.ForeignKey(Model, related_name='det_model', on_delete=models.CASCADE)

    class PreProcess(models.TextChoices):
        DEFAULT = 'DE', _('Default')
        MTCNN = 'MT', _('MTCNN')
        SPHERE = 'SP', _('SphereFace')
        OPEN = 'OP', _('OpenFace')
    ret_pre_process = models.CharField(max_length=2, choices=PreProcess.choices)
    save_path = models.CharField(max_length=200, default='/home')

    last_update = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        ordering = ['-last_update']


class Database(models.Model):
    name = models.CharField(max_length=200)
    quantity = models.IntegerField(default=0)
    last_update = models.DateTimeField(auto_now_add=True)
    feature_mean = models.CharField(max_length=12000, default='0')

    def __str__(self):
        return self.name


class ImageDB(models.Model):
    operation = models.ForeignKey(Operation, on_delete=models.PROTECT)
    database = models.ForeignKey(Database, on_delete=models.PROTECT)

    path = models.CharField(max_length=200)
    bb = models.CharField(max_length=100)
    features = models.CharField(max_length=12000)
    label = models.CharField(max_length=100)


class Processed(models.Model):
    operation = models.ForeignKey(Operation, on_delete=models.CASCADE)

    path = models.CharField(max_length=200)
    frame = models.IntegerField()


class FullProcessed():  
    def __init__(self, path="", operation=0, detection_result_path = ""):
        self.operation = operation
        self.path = path
        self.detection_result_path = detection_result_path
        self.detections = []
        self.faces = []

    class Detection():
        label_to_superlabel = {'pistol': 'Arma de Fogo',
                               'knife': 'Arma Branca',
                               'bicycle': 'Veículo',
                               'car': 'Veículo',
                               'bus': 'Veículo',
                               'motorbike': 'Veículo',
                               'truck': 'Veículo',
                               'machete': 'Arma Branca',
                               'pocket/stiletto knife': 'Arma Branca',
                               'axe': 'Arma Branca',
                               'rifle/sniper': 'Arma de Fogo',
                               'shotgun': 'Arma de Fogo',
                               'machine gun': 'Arma de Fogo',
                               'submachine gun': 'Arma de Fogo',
                              }

        def __init__(self, label, score, bbx):
            self.label = self.label_to_superlabel[label]
            self.score = score * 100.0
            self.bbx = bbx
            

    class Faces():
        class Ranking():
            def __init__(self, position, value, imgdb):
                self.position = position
                self.value = value
                self.imgdb = imgdb

        def __init__(self, id, bbx):
            self.id = id
            self.bbx = bbx
            self.rankings = []

    

class Output(models.Model):
    processed = models.ForeignKey(Processed, on_delete=models.CASCADE)
    obj = models.IntegerField(default=0)

    class ParameterOpt(models.TextChoices):
        BB = 'BB', _('Bounding Box')
        SCORE = 'SC', _('Score')
        LABEL = 'LB', _('Label')
    parameter = models.CharField(max_length=2, choices=ParameterOpt.choices)

    value = models.CharField(max_length=100)


class Ranking(models.Model):
    processed = models.ForeignKey(Processed, on_delete=models.CASCADE)
    imagedb = models.ForeignKey(ImageDB, on_delete=models.CASCADE)

    position = models.IntegerField()
    value = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=['processed_id', 'imagedb_id']),
        ]
        constraints = [
            models.UniqueConstraint(fields=['processed_id', 'imagedb_id'], name='ranking_ids')
        ]
