ROOT=$PWD

rm $ROOT/features.mat
for people in "Winona_Ryder" "John_Travolta" "JK_Rowling" "Nelson_Mandela" "Zico"
do
	for i in 1 2 3
	do
		echo Adding image $i of $people in a dataset.
  		python3 -W ignore main.py --operation extract_features \
					    		  --feature_file $ROOT/features.mat \
					    		  --path_image_query $ROOT/datasets/LFW/lfw/$people/$people"_000"$i.jpg \
					    		  --query_label $people \
					    		  --model_name mobilefacenet\
					    		  --preprocessing_method sphereface

  	done
done

echo Generating rankign for image 4 of Winona_Ryder.

python3 -W ignore main.py --operation generate_rank \
					      --feature_file $ROOT/features.mat \
					      --path_image_query $ROOT/datasets/LFW/lfw/Winona_Ryder/Winona_Ryder_0004.jpg \
					      --model_name mobilefacenet\
					      --preprocessing_method sphereface

rm $ROOT/features.mat


echo Extrating features of the LFW dataset.

python3 -W ignore main.py --operation extract_features \
		    			  --feature_file $ROOT/features.mat \
		    			  --dataset LFW \
		    			  --specific_dataset_folder_name lfw \
		    			  --model_name mobilefacenet\
		    			  --preprocessing_method sphereface

echo Generating ranking of the whole dataset. 

python3 -W ignore main.py --operation generate_rank \
					      --feature_file $ROOT/features.mat \
		    			  --dataset LFW \
		    			  --specific_dataset_folder_name lfw \
		    			  --model_name mobilefacenet\
		    			  --preprocessing_method sphereface

echo Generating rankign for image 4 of Winona_Ryder of the whole dataset.

python3 -W ignore main.py --operation generate_rank \
					      --feature_file $ROOT/features.mat \
					      --path_image_query $ROOT/datasets/LFW/lfw/Winona_Ryder/Winona_Ryder_0004.jpg \
					      --model_name mobilefacenet\
					      --preprocessing_method sphereface

echo Extracting and generating ranking for the whole dataset and save the image results.

python3 -W ignore main.py --operation extract_generate_rank \
						  --result_sample_path $ROOT/results/ \
		    			  --dataset LFW \
		    			  --specific_dataset_folder_name lfw \
		    			  --img_extension jpg \
		    			  --model_name mobilefacenet\
		    			  --preprocessing_method sphereface

rm $ROOT/features.mat


for dataset in "LFW" "YALEB"
do 
	if [ "$dataset" = "LFW" ]
	then
		specific_dataset_folder_name=lfw
		img_extension=jpg
	else
		specific_dataset_folder_name=ExtendedYaleB
		img_extension=pgm
	fi
	echo Extrating features and generating ranking of $dataset dataset.

	python3 -W ignore main.py --operation extract_generate_rank \
			    			  --dataset $dataset \
			    			  --specific_dataset_folder_name $specific_dataset_folder_name \
			    			  --img_extension $img_extension \
			    			  --model_name mobilefacenet\
			    			  --preprocessing_method sphereface
done 

echo Extrating features without using query_label --expecting assert--

python3 -W ignore main.py --operation extract_features \
		    		      --feature_file $ROOT/features.mat \
		    		      --path_image_query $ROOT/datasets/LFW/lfw/$Winona_Ryder/Winona_Ryder_0001.jpg \
		    		      --model_name mobilefacenet \
		    		      --preprocessing_method sphereface
		    		      # --query_label $people \
