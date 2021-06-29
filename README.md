# mp_e04

## Docker

```buildoutcfg
docker build -t mp/e04:v2 .
```

```buildoutcfg
docker run -it -d --name keiller_mp_3 --gpus all --shm-size 20G --mount type=bind,source=/mnt/DADOS_PONTOISE_1/keiller,destination=/mnt/DADOS_PONTOISE_1/keiller mp/e04:v2 /bin/bash
```

## Face Recognition

### train()

Method definition:
```python
train(dataset_path, save_dir, resume_path=None)
```
where:

1. ```dataset_path``` is the path to the dataset used to train.
2. ```save_dir``` is the path to the dir used to save the trained model.
3. ```resume_path``` is the path to a previously trained model (for fine-tuning)

Test call:

```python
python -W ignore train.py
```

### retrieval()

Method definition:
```
retrieval(image_path, feature_file, save_dir, method="image")
```
where:

1. ```image_path``` is the path to the analysed image.
2. ```feature_file``` is the path to the file that contains extracted features from dataset images.
3. ```save_dir``` is the path to the dir used to save the results.
4. ```method``` is the method to export the results, json or image.

Test call:

```
python -W ignore retrieval.py --image_path x --feature_file y --save_dir z --method image 
```

### The following functions are inside ```manipulate_dataset.py```

### update_dataset()

Method definition:
```
update_dataset(image_path, save_dir, img_ID, feature_file)
```
where:

1. ```image_path``` is the path to the analysed image.
2. ```save_dir``` is the path to the dir used to save the results.
3. ```img_ID``` is the name of the person in the analysed image.
4. ```feature_file``` is the path to a feature file that already exists, if there is any (add image info to an existing file), or the path to create a new feature file, if there is no existing feature file in that path.

Test call:

```python
python -W ignore manipulate_dataset.py --operation update_dataset --save_dir w --image_path x --image_id y --feature_file z
```
### create_dataset()

Method definition:
```
create_dataset(save_dir, feature_file, image_path=None, dataset=None, dataset_path=None)
```
where:

1. ```save_dir``` is the path to the dir used to save the results.
2. ```feature_file``` is the path to a feature file that already exists, if there is any (add image info to an existing file), or the path to create a new feature file, if there is no existing feature file in that path.
3. ```image_path``` is the path to the analysed image.
4. ```dataset``` is the name of the analysed dataset.
5. ```dataset_path``` is the path to the analysed dataset.

Test call:

If a dataset is being passed:
```python
python -W ignore manipulate_dataset.py --operation create_dataset --dataset w --dataset_path x --save_dir y --feature_file z
```
If a single image is being passed:
```python
python -W ignore manipulate_dataset.py --operation create_dataset --image_path x --save_dir y --feature_file z
```

## Object Detection
Possui duas funções `train` e `retrieval`

### train_light()
Method definition:
```
def train_light(hyp_path, data, output_path)
```
where:

1. ```hyp_path``` is the path to the hyperparameters .yaml file.
2. ```data``` is the path to the dataset .yaml file.
3. ```output_path``` is the path to the directory to save the train results.
Test call:
```
python train_light.py --data path_to_dataset.yaml
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

