# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mp.settings")

import django
django.setup()
from e04.models import Operation, Model, GeneralConfig
from django.contrib.auth.models import User


def create_admin_user():
    print('Creating admin user...')
    user = User.objects.create_user('admin', '', 'admin', is_superuser=True, is_staff=True)
    return user


def create_operation(user):
    operation = Operation()
    operation.user = user
    operation.type = Operation.OpType.TRAIN
    operation.status = Operation.OpStatus.FINISHED
    operation.save()
    return operation


def create_model(operation, type, name, model_path, val_acc):
    model = Model()
    model.op_id = operation

    model.type = type
    model.name = name
    model.model_path = model_path
    model.val_acc = val_acc
    model.save()
    return model


def create_gen_config(user, model_det, model_ret):
    gen_c = GeneralConfig()
    gen_c.ret_model = model_ret
    gen_c.det_model = model_det
    gen_c.ret_pre_process = GeneralConfig.PreProcess.SPHERE
    gen_c.save_path = "/mnt/DADOS_PONTOISE_1/keiller/mp_e04/frontend/mp/results/"
    gen_c.user = user
    gen_c.save()


def main():
    # create default user for the system
    user = create_admin_user()

    # create first operations
    op_train_det = create_operation(user)
    op_train_ret = create_operation(user)

    # populate models
    model_det = create_model(op_train_det, Model.ModelType.OBJECT, "YoloV5L6",
                             "/mnt/DADOS_PONTOISE_1/keiller/mp_e04/frontend/mp/models/yolov5l6.pt", 78.72)
    model_ret = create_model(op_train_ret, Model.ModelType.FACE, "curricularface", "", 97.48)

    # create general configuration
    create_gen_config(user, model_det, model_ret)


if __name__ == "__main__":
    main()

