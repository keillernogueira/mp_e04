# Projeto RecFaces

## Dependências

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [Scipy](https://www.scipy.org/install.html#installing-via-pip)
- [ImageIO](https://imageio.readthedocs.io/en/stable/installation.html)
- [SKLearn](https://scikit-learn.org/stable/install.html)
- [OpenCV Python](https://pypi.org/project/opencv-python/)
- [dlib](https://pypi.org/project/dlib/)
- [py7zr](https://pypi.org/project/py7zr/)

## Usage

Para testar um modelo (já treinado) basta executar o comando:

```
python eval_retrieval.py --operation OPERATION
                         [--feature_file FEATURE_FILE]
                         [--result_sample_path RESULT_SAMPLE_PATH]
                         [--path_image_query PATH_IMAGE_QUERY]
                         [--query_label QUERY_LABEL]
                         [--dataset DATASET_NAME]
                         [--specific_dataset_folder_name DATASET_FOLDER_NAME]
                         [--img_extension EXTENSION]
                         [--preprocessing_method PRE_PROCESS_NAME]
                         --model_name MODEL_NAME
                         [--batch_size BATCH_SIZE]
```
onde, 
1. ```operation``` é a operação que será realizada pelo algoritmo
    1. parâmetro **requerido**
    2. opções: 
        1. ```extract_features```, para extrair features de uma imagem ou dataset
        1. ```generate_rank```, para gerar o ranking de uma imagem ou dataset
        1. ```extract_generate_rank```, para extrair features e avaliar o dataset todo. 
        **SOMENTE É VÁLIDO PARA O PROCESSAMENTO DE DATASETS.**
1. ```feature_file``` é nome do arquivo (com caminho e extensão) de onde as features serão carregadas ou salvas
    1. parâmetro opcional
    2. default: ```None```
    3. extensão do arquivo: ```.mat```
    4. exemplo: ```result_features/features.mat```
1. ```result_sample_path``` é um caminho (path) para salvar os plots com exemplos dos resultados
    1. parâmetro opcional
    2. defaut: ```None```
    3. Se for setado como um path válido, ao processar um dataset, imagens de exemplo serão salvas nesse caminho. 
    **SOMENTE É VÁLIDO PARA O PROCESSAMENTO DE DATASETS.**
1. ```path_image_query``` é um caminho (path) para a imagem query
    1. parâmetro opcional
    2. defaut: ```None```
    3. Se for setado como um path válido, somente métodos relacionados ao processamento de imagens serão usados.
    Métodos relacionados ao processamento de datasets não serão usados neste caso.
1. ```query_label``` label ou ID da query
    1. parâmetro opcional
    2. defaut: ```None```
    3. É requerido no caso de adicionar uma nova imagem ao database.
1. ```dataset``` é o nome do dataset que terá as features extraídas
    1. parâmetro opcional
    2. defaut: ```None```
    3. opções:
        1. ```LFW```
        1. ```YALEB```
    4. Se for setado como um valor válido, somente métodos relacionados ao processamento de datasets serão usados.
    Métodos relacionados ao processamento de imagens simples não serão usados neste caso.
1. ```specific_dataset_folder_name``` é a pasta específica do dataset
    1. parâmetro opcional
    2. defaut: ```None```
    3. opções poderiam incluir ```lfw``` ou ```ExtendedYaleB```, por exemplo
    4. Se a flag ```dataset``` for usada, esse parâmetro passa a ser obrigatório.
    5. **eventualmente, essa paramêtro será retirado**
1. ```img_extension``` é a extensão das imagens desse dataset
    1. parâmetro opcional
    2. defaut: jpg
    2. opções comuns incluem:
        1. ```jpg``` (para LFW)
        1. ```pgm``` (para YALEB)
        1. ```txt``` (para imagens base64)
1. ```preprocessing_method``` é método de pre-processamento a ser utilizado nas imagens
    1. parâmetro opcional
    2. defaut: ```None```
    3. opções:
        1. ```mtcnn```
        1. ```sphereface```
        1. ```openface```
        1. ```None``` (nessa opção, é feito um crop no centro da imagem por padrão)
    4. Se algum método de pre-processamento falhar, o método default ```None``` é executado.
1. ```model_name``` é o nome do modelo que será usado na extração de *features*
    1. parâmetro **requerido**
    2. opções: 
        1. ```mobilefacenet```
        1. ```mobiface```
        1. ```sphereface```
        1. ```openface```
        1. ```facenet```
        1. ```shufflefacenet```
1. ```batch_size``` é o tamanho do *batch size* usado para extrair as *features*
    1. parâmetro opcional
    2. defaut: 32
    3. para dataset YALEB, por motivos de mémoria de GPU, default: 8

## Métodos Integrados

Criados **COM** foco mobile:
1. MobileFaceNet (2018)
    1. [artigo](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)
    2. [repo](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
2. MobiFace (2018)
    1. [artigo](https://arxiv.org/pdf/1811.11080.pdf)
    2. Implementação própria do artigo
    
Criados **SEM** foco mobile:
1. FaceNet (2015)
    1. [artigo](https://arxiv.org/pdf/1503.03832.pdf)
    2. [repo](https://github.com/timesler/facenet-pytorch)
1. OpenFace (2016)
    1. [artigo](http://elijah.cs.cmu.edu/DOCS/CMU-CS-16-118.pdf)
    2. [repo_v1](https://github.com/TwoBranchDracaena/OpenFace-PyTorch)
    3. [repo_v2](https://github.com/thnkim/OpenFacePytorch) (usado)
1. SphereFace (2017)
    1. [artigo](https://arxiv.org/pdf/1704.08063.pdf)
    2. [repo](https://github.com/clcarwin/sphereface_pytorch)
1. SuffleFaceNet (2019)
    1. [artigo](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf)
    2. Implementação própria do artigo

## Datasets

Para treinar:
 
  - [CASIA-Webface](https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz) (default)
  - [MS-Celeb-1M](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/)

Para testar:

  - [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
  - [Yale Face Database B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)

## Resultados

  | Method | Dataset | Pre-Processing | mAP | AP@1 | AP@5 | AP@10 | AP@20 | AP@50 | AP@100 |
  |:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | LFW | None  | 39.96 | 88.32 | 76.65 | 67.77 | 55.86 | 44.75 | 41.17 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | LFW | MTCNN | 77.18 | 97.29 | 94.62 | 92.21 | 86.28 | 79.11 | 77.65 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | LFW | SphereFace | 91.61 | 96.76 | 96.52 | 96.42 | 95.73 | 92.69 | 92.10 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | LFW | OpenFace | 38.04 | 83.09 | 70.05 | 61.47 | 51.47 | 41.80 | 38.65 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | YaleB | None | 77.03 | 99.99 | 99.93 | 99.77 | 99.27 | 97.60 | 94.13 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | YaleB | MTCNN | 37.26 | 97.68 | 93.13 | 88.70 | 82.86 | 73.78 | 66.35 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | YaleB | SphereFace | 57.44 | 98.60 | 95.35 | 92.29 | 88.44 | 83.00 | 79.41 |
  | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | YaleB | OpenFace |16.01 | 97.09 | 90.78 | 84.14 | 74.80 | 58.55 | 43.28 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | LFW | None  | 15.92 | 48.59 | 31.90 | 25.45 | 20.20 | 15.68 | 13.85 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | LFW | MTCNN | 33.10 | 76.44 | 63.66 | 55.78 | 46.30 | 37.41 | 34.55 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | LFW | SphereFace | 87.84 | 95.43 | 94.49 | 94.07 | 92.73 | 88.86 | 88.39 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | LFW | OpenFace | 24.99 | 68.64 | 53.56 | 45.59 | 37.50 | 29.42 | 26.19 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | YaleB | None | 57.22 | 99.98 | 99.87 | 99.52 | 98.42 | 94.09 | 86.34 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | YaleB | MTCNN | 17.49 | 93.97 | 84.18 | 76.06 | 66.60 | 52.76 | 41.14 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | YaleB | SphereFace | 46.05 | 96.91 | 91.50 | 86.42 | 80.81 | 73.88 | 69.47 |
  | [SphereFace](https://github.com/clcarwin/sphereface_pytorch) | YaleB | OpenFace | 13.67 | 97.50 | 90.03 | 81.08 | 69.54 | 52.01 | 37.42 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | LFW | None | 78.51 | 97.05 |94.46 | 91.58 | 85.54 | 79.27 | 78.54 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | LFW | MTCNN | 97.42 | 99.86 |99.85 | 99.82 | 99.41 | 97.61 | 97.49 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | LFW | SphereFace | 82.76 | 96.22 |94.92 | 93.69 | 89.67 | 83.84 | 83.14 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | LFW | OpenFace | 68.11 | 93.28 | 88.73 | 84.75 | 77.38 | 69.88 | 68.56 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | YaleB | None | 33.07 | 98.53 |94.20 | 90.05 | 84.25 | 74.52 | 65.03 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | YaleB | MTCNN | 53.96 | 97.94 |93.98 | 90.60 | 86.45 | 80.80 | 76.90 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | YaleB | SphereFace | 33.84 | 96.93 |91.05 | 86.24 | 80.20 | 71.16 | 63.80 |
  | [FaceNet](https://github.com/timesler/facenet-pytorch) | YaleB | OpenFace | 23.33 | 97.57 | 91.98 | 86.33 | 78.41 | 65.85 | 54.87 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | LFW | None | 29.47 | 67.74 |53.72 | 46.48 | 38.49 | 30.74 | 28.55 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | LFW | MTCNN | 25.90 | 65.89 |51.43 | 43.81 | 35.40 | 28.05 | 25.54 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | LFW | SphereFace | 54.01 | 87.82 |81.19 | 75.83 | 67.35 | 58.55 | 56.21 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | LFW | OpenFace | 13.33 | 37.24 | 23.61 | 18.69 | 14.65 | 11.21 | 10.04 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | YaleB | None | 12.09 | 86.21 |70.72 | 61.17 | 50.17 | 35.05 | 24.73 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | YaleB | MTCNN | 12.33 | 75.08 |58.71 | 50.49 | 42.35 | 32.08 | 24.20 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | YaleB | SphereFace | 23.07 | 85.44 |73.66 | 67.44 | 61.11 | 52.82 | 45.95 |
  | [MobiFace](https://arxiv.org/pdf/1811.11080.pdf) | YaleB | OpenFace | 07.37 | 75.11 | 53.40 | 41.74 | 30.76 | 18.83 | 12.16 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | LFW | None | 11.88 | 35.65 | 22.39 | 17.22 | 12.75 | 09.24 | 08.05 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | LFW | MTCNN | 11.31 | 32.64 | 20.00 | 15.29 | 11.33 | 08.10 | 07.14 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | LFW | SphereFace | 17.03 | 48.46 | 32.80 | 26.44 | 20.71 | 16.02 | 14.37 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | LFW | OpenFace | 52.09 | 82.73 | 73.68 | 67.41 | 59.13 | 51.94 | 51.16 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | YaleB | None | 09.74 | 86.70 | 70.25 | 58.46 | 44.41 | 27.22 | 17.56 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | YaleB | MTCNN | 06.75 | 67.34 | 45.91 | 35.35 | 25.71 | 15.47 | 09.79 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | YaleB | SphereFace | 10.39 | 75.12 | 58.01 | 49.20 | 40.20 | 28.54 | 20.28 |
  | [OpenFace](https://github.com/thnkim/OpenFacePytorch) | YaleB | OpenFace | 09.74 | 86.70 | 70.25 | 58.46 | 44.41 | 27.22 | 17.56 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | LFW | None | 85.61 | 98.24 | 96.94 | 95.77 | 92.08 | 86.65 | 86.04 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | LFW | MTCNN | 77.17 | 96.39 | 93.52 | 91.05 |  85.09 | 78.21 | 77.41 | 
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | LFW | SphereFace | 67.41 | 93.58 | 89.50 | 85.46 | 77.61 | 69.51 | 67.87 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | LFW | OpenFace | 16.15 | 55.70 | 38.97 | 31.12 | 23.61 | 16.86 | 14.37 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | YaleB | None | 19.33 | 97.46 | 91.09 | 85.57 | 78.14 | 62.74 | 47.20 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | YaleB | MTCNN | 16.76 | 91.64 | 80.65 | 73.34 | 64.95 | 52.17 | 40.63 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | YaleB | SphereFace | 29.07 | 96.13 | 88.86 | 83.62 | 77.30 | 68.00 | 60.49 |
  | [ShuffleFaceNet](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf) | YaleB | OpenFace | 11.40 | 94.91 | 83.90 | 74.28 | 61.59 | 41.97 | 28.05 |
  
 ## Datasets construídos
 
 1. Patreo5
     1. 5 pessoas; 15 fotos por pessoa; 75 fotos no total
     2. Imagens capturadas em 1 dia(roupas iguais)
     3. 1 câmera fotográfica
 2. Patreo3
     1. 3 pessoas; 15 fotos por pessoa; 45 fotos no total
     2. Imagens capturadas em 3 dias diferentes
     3. 5 câmeras fotográficas
     
 | Method | Dataset | Pre-Processing | mAP | AP@1 | AP@5 | AP@10 | AP@20 | AP@50 | AP@100 |
 |:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo3 | SphereFace  | 81.78 | 91.11 | 85.37 | 84.35 | 72.70 | 81.78 | 81.78 |
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo5 | SphereFace  | 81.58 | 90.66 | 88.92 | 84.92 | 75.27 | 80.31 | 81.58 |
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo3 + Patreo5 | SphereFace  | 79.62 | 90.83 | 88.78 | 85.78 | 80.15 | 74.83 | 78.81 |
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo3 + LFW | SphereFace  | 91.12 | 96.64 | 96.27 | 96.06 | 95.17 | 92.17 | 91.60 | 
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo5 + LFW | SphereFace  | 91.01 | 96.51 | 96.17 | 95.98 | 95.00 | 92.05 | 91.49 |
 | [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | Patreo3 + Patreo5 + LFW | SphereFace  | 90.68 | 96.53 | 96.04 | 95.85 | 94.89 | 91.62 | 91.12 |

## Utils

1. [Repo Face Recognition PyTorch](https://github.com/grib0ed0v/face_recognition.pytorch)
2. [Face evoLVe PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#introduction)
3. [Image Retrieval by Fine-Tuning CNN](https://github.com/layumi/Image-Retrieval-by-Finetuning-CNN)
4. [Deep Face Image Retrieval: a Comparative Study with Dictionary Learning](https://arxiv.org/pdf/1812.05490.pdf)
