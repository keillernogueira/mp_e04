
# 1 teste main.py

## dataset_processor: extrair features e avaliar um dataset inteiro

	CUDA_VISIBLE_DEVICES=0 python main.py --operation extract_features --feature_file test.mat --dataset LFW --specific_dataset_folder_name lfw --img_extension jpg --model_name mobilefacenet --preprocessing_method sphereface --result_sample_path outputs/
	# necessario a flag result_sample_path para salvar as classes

	CUDA_VISIBLE_DEVICES=0 python main.py --operation generate_rank --feature_file test.mat --dataset LFW --specific_dataset_folder_name lfw --img_extension jpg --model_name mobilefacenet --preprocessing_method sphereface

## image_processor: gerar ranking de uma imagem e extrair featues de uma imagem

	CUDA_VISIBLE_DEVICES=0 python3 main.py --operation generate_rank --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0028.jpg --model_name mobilefacenet --preprocessing_method sphereface

	CUDA_VISIBLE_DEVICES=0 python3 main.py --operation extract_features --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0028.jpg --query_label Luiz_Inacio_Lula_da_Silva --model_name mobilefacenet --preprocessing_method sphereface

## image_processor: gerar ranking de uma imagem com multiplas faces

	CUDA_VISIBLE_DEVICES=0 python main.py --operation generate_rank --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0004.jpg --model_name mobilefacenet --preprocessing_method sphereface
	
## video_processor: processar video

	CUDA_VISIBLE_DEVICES=0 python main.py --operation generate_rank --feature_file test.mat --video_path https://www.youtube.com/watch?v=-TXBxxPAtb0 --model_name mobilefacenet --preprocessing_method sphereface

--------------------------------------------------------------------------------------------------------------

# 2 teste train.py

## com path default

	CUDA_VISIBLE_DEVICES=0 python train.py --dataset_path datasets/CASIA-WebFace/ --save_dir outputs/ --num_epoch 6

## com path customizado

	CUDA_VISIBLE_DEVICES=0 python train.py --dataset_path datasets/CASIA-WebFace/ --save_dir outputs/ --num_epoch 6 --resume_path outputs/005.ckpt

--------------------------------------------------------------------------------------------------------------

# 3 teste retrieval.py

## com path default - output_method image

	CUDA_VISIBLE_DEVICES=0 python retrieval.py --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0001.jpg --model_name mobilefacenet --preprocessing_method sphereface --save_dir outputs/

## com path default - output_method json

	CUDA_VISIBLE_DEVICES=0 python retrieval.py --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0001.jpg --output_method json --model_name mobilefacenet --preprocessing_method sphereface --save_dir outputs/

## com path customizado

	CUDA_VISIBLE_DEVICES=0 python retrieval.py --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0001.jpg --model_name mobilefacenet --preprocessing_method sphereface --save_dir outputs/ --model_path outputs/005.ckpt
	
## com multiplas faces

	CUDA_VISIBLE_DEVICES=0 python retrieval.py --feature_file test.mat --image_query datasets/LFW/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0004.jpg --model_name mobilefacenet --preprocessing_method sphereface --save_dir outputs/

## com input json
	
	CUDA_VISIBLE_DEVICES=0 python retrieval.py --feature_file test.mat --data_to_process exemplos/input_json.json --output_method json --model_name mobilefacenet --preprocessing_method sphereface --save_dir outputs/

--------------------------------------------------------------------------------------------------------------

# 4 teste manipulate_dataset.py

## com uma imagem

	CUDA_VISIBLE_DEVICES=0 python manipulate_dataset.py --feature_file /mnt/DADOS_PARIS_1/keiller/mp_e04/pessoas/test.mat --dataset_path /mnt/DADOS_PARIS_1/keiller/mp_e04/pessoas/datasets/TESTE/

## com um dataset

	CUDA_VISIBLE_DEVICES=0 python manipulate_dataset.py --feature_file /mnt/DADOS_PARIS_1/keiller/mp_e04/pessoas/test.mat --dataset_path /mnt/DADOS_PARIS_1/keiller/mp_e04/pessoas/datasets/LFW/lfw/


--------------------------------------------------------------------------------------------------------------
