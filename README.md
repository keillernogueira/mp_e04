# mp_e04

## Face Recognition

### train()

Method definition:
```
train(dataset_path, save_dir, resume_path=None)
```
where:

1. ```dataset_path``` is the path to the dataset used to train.
2. ```save_dir``` is the path to the dir used to save the trained model.
3. ```resume_path``` is the path to a previously trained model (for fine-tuning)

Test call:

```
python -W ignore train.py
```

### retrieval()

Method definition:
```
retrieval(image_path)
```
where:

1. ```image_path``` is the path to the analysed image.

Test call:

```
python retrieval.py
```

### update_dataset()

Method definition:
```
update_dataset(img_path, img_ID, feature_file = "features.mat")
```
where:

1. ```img_path``` is the path to the analysed image.
2. ```img_ID``` is the name of the person in the analysed image.
3. ```feature_file``` is the path to a feature file that already exists, if there is any (add image info to an existing file), or the path to create a new feature file, if there is no existing feature file in that path.

Test call:

```
python update_dataset.py
```

## Object Detection
Possui duas funções `train` e `retrieval`

### train()
```
def train(dataset_path, **kwargs):
    - fazer um tratamento de parametros similar ao da __main__ em https://github.com/ultralytics/yolov5/blob/master/train.py

    - tratamento dos **kwargs (parametros de treino: numero de epocas, learning_rate, etc)
        - criar o arquivo .yaml dos hyperparametros de treino se nao existir

    - processar (ou preprocessar) o dataset deixando no formato da yolo caso nao esteja (opcional, vamos assumir que esteja)
        - criar o arquivo .yaml que eles usam se nao existir

    - chamar a função pronta da yolov5 para treino: train(hyp, opt, device, logger)
```

### detect_obj()
```
def detect_obj(img_path, model_path, output_path, format):
    - abrir imagem (seja local, ou por download)
    
    - inferencia usando o modelo salvo
    
    - gerar a saida de acordo com o formato
```

Test call:

```
python detect_obj.py --weights path_to_model --source path_to_target_images --format format_output (json, images or both) --output-path output_folder_path
```
- The `weights` parameter can be yolov5m.pt, that will be downloaded if necessary (or any weight in this [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints)) to pretrained COCO model, will detect objectcs from the 80 classes on COCO).
- The `source` parameter can be a folder with image/video files, a single file (local or url), or a json with a list of files. The files url can also be youtube links.

