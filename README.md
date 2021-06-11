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
