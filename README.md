# MPMG: E04

## Resultados

[Planilha](https://docs.google.com/spreadsheets/d/1-3UkWuNJaoVz_6OtG2eQEH1DZFWOrG2IF4uQoskUsjI/edit?usp=sharing)

## Docker

Criar imagem docker usando Dockerfile (este comando precisa ser executado na mesma pasta do Dockerfile):

```
docker build -t mp/e04:latest .
```
```-t``` é a tag da imagem

Exemplo de como rodar um comando python (para retrieval) dentro de container docker: 

```
docker run --name mp_e04_t1 --gpus all --ipc=host --rm -it -v VOLUME_HOST:VOLUME_DOCKER mp/e04:latest python PATH/retrieval.py --feature_file PATH/features.mat --image_query PATH/IMAGE.jpg --model_name mobilefacenet --preprocessing_method sphereface --save_dir PATH/outputs/
```

## Reconhecimento de pessoas

### train.py

Algoritmo usado para treinar uma rede.

```python
train(dataset_path, save_dir, resume_path=None, num_epoch=71)
```
onde:

1. ```dataset_path``` é caminho/path para o dataset. O dataset deve ser uma pasta com uma subpasta para cada pessoa 
   (com o nome da pasta sendo o nome da pessoa). Dentro dessa pasta, podem haver várias imagens daquela pessoa.
2. ```save_dir``` é pasta onde será salvo o modelo treinado.
3. ```resume_path``` é um caminho para carregar um modelo pre-treinado.
   1. Default: ```None```
   2. Se não for setado (ou seja, se ficar ```None```), o algoritmo carrega um modelo já pré-treinado com o dataset
    [CASIA Web-Face](https://arxiv.org/abs/1411.7923).
4. ```num_epoch```, número de épocas que será usado para treinar o modelo
   1. Default: ```71```
   
OBS.: Esse função assuma que um GPU está disponível para o treino.

### manipulate_dataset.py

Algoritmo usado para manipular (adicionar _features_) a um dataset.

```python
manipulate_dataset(feature_file, dataset_path, model_name="mobilefacenet", model_path=None, 
                   preprocessing_method="sphereface", crop_size=(96, 112), gpu=True)
```
where:

1. ```feature_file``` é o caminho para um arquivo (```.mat```) de _features_.
   Se esse arquivo já existir, adiciona novas _features_ extraídas nesse arquivo.
   Caso contrário, cria o arquivo com as novas _features_ extraídas. 
2. ```dataset_path``` caminho para o dataset. O formato do dataset é o mesmo [já descrito anteriormente](#trainpy).
3. ```model_name``` é o nome do modelo a ser usado.
   1. Default: ```mobilefacenet```
4. ```model_path``` é um caminho para carregar um modelo pre-treinado.
   1. Default: ```None```
   2. Se não for setado (ou seja, se ficar ```None```), o algoritmo carrega um modelo já pré-treinado com o dataset
    [CASIA Web-Face](https://arxiv.org/abs/1411.7923).
5. ```preprocessing_method``` é o algoritmo de pre-processamento a ser usado para detectar as faces.
   1. Default: ```sphereface```
6. ```crop_size``` é o tamanho do _crop_ a ser feito em volta da face detectada.
   1. Default: ```(96, 112)```
7. ```gpu``` flag que indica se GPU será usada no processamento.
   1. Default: ```True```

### retrieval.py

Algoritmo usado para fazer _retrieval_.

```
retrieval(image_path, feature_file, save_dir, output_method="image", model_name="mobilefacenet", model_path=None,
          preprocessing_method="sphereface", crop_size=(96, 112), gpu=True)
```
where:

1. ```image_path``` é o caminho para a imagem _query_.
   1. Esse arquivo pode ser uma imagem ou um json no formato base64 ([exemplo](exemplos/image_base64.json)).
2. ```feature_file``` é o caminho para um arquivo (```.mat```) de _features_ já extraídas.
3. ```save_dir``` é pasta onde será salvo o ranking de saída.
4. ```output_method``` é o método que será usada para exportar a saída
   1. Duas opções: ```json``` ([exemplo](exemplos/saida.json)) ou ```image```.
4. ```model_name``` é o nome do modelo a ser usado.
   1. Default: ```mobilefacenet```
5. ```model_path``` é um caminho para carregar um modelo pre-treinado.
   1. Default: ```None```
   2. Se não for setado (ou seja, se ficar ```None```), o algoritmo carrega um modelo já pré-treinado com o dataset
    [CASIA Web-Face](https://arxiv.org/abs/1411.7923).
6. ```preprocessing_method``` é o algoritmo de pre-processamento a ser usado para detectar as faces.
   1. Default: ```sphereface```
7. ```crop_size``` é o tamanho do _crop_ a ser feito em volta da face detectada.
   1. Default: ```(96, 112)```
8. ```gpu``` flag que indica se GPU será usada no processamento.
   1. Default: ```True```

## Object Detection

### train_light.py

Algoritmo usado para treinar um modelo.

```
train(hyp_path='hyp.scratch.yaml', data='dataset.yaml', output_path='runs/train/exp', 
      opt=defaultOptTrain())
```
onde:

1. ```hyp_path``` é o caminho para um arquivo ```yaml``` com a configuração de hiperparâmetros.
   1. Detault: ```hyp.scratch.yaml```.
2. ```data``` é o caminho para um arquivo ```yaml``` com a configuração do dataset.
   1. Detault: ```dataset.yaml``` (usa o dataset OD-Weapon)
3. ```output_path``` é o caminho para salvar o modelo treinado.
4. ```opt``` é um arquivo que carrega as opções (_flags_) passadas para o algoritmo.
   1. use a flag ```--help``` para saber mais das opções

### detect_obj.py

Algoritmo usado para detectar objetos.

```
retrieval(img_path, model_path, output_path, save_as, opt=defaultOpt(), output_file='detections.json')
```
onde:

1. ```img_path``` é o caminho para a imagem ou vídeo
   1. Pode ser uma pasta com arquivos de imagem/vídeo, um único arquivo (local ou url) 
      ou um [json](exemplos/input_object.json) com uma lista de arquivos. O url dos arquivos também pode ser links do YouTube. 
2. ```model_path``` é um caminho para carregar um modelo pre-treinado.
   1. Default: ```yolov5s.pt```
3. ```output_path``` é o caminho para salvar a saída do algoritmo.
4. ```save_as``` é o formato em que será salva a saída.
   1. Opções: ```img``` ou ```json``` 
   1. Default: ```img```
5. ```opt``` é um arquivo que carrega as opções (_flags_) passadas para o algoritmo.
   1. use a flag ```--help``` para saber mais das opções
6. ```output_file``` é o nome do arquivo de saída.
   1. Somente usado para o ```--save_as json```
   2. Default: ```detections.json```
   3. [Exemplo](exemplos/detections9.json)
