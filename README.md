# mp_e04

## Face Recognition

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

### retrieval()
```
def retrieval(img_path, model_path, output_path, format):
    - abrir imagem (seja local, ou por download)
    
    - inferencia usando o modelo salvo
    
    - gerar a saida de acordo com o formato
```
