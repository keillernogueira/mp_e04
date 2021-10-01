from utils import *
from processors.dataset_processor import process_dataset
from processors.image_processor import process_image
from processors.video_processor import process_video

from networks.load_network import load_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: extract_features | generate_rank |'
                             ' extract_generate_rank (for dataset only)]')
    parser.add_argument('--feature_file', type=str, default=None,
                        help='File path to save/load the extracted features (.mat file).')
    parser.add_argument('--result_sample_path', type=str, default=None,
                        help='Path to save image samples of the results.')
    parser.add_argument('--gpu', type=str2bool, default=True,
                        help='Boolean to indicate the use of GPU.')

    # image processing options
    parser.add_argument('--image_query', type=str, default=None,
                        help='Path or string of query image. If set, only this image is processed.')
    parser.add_argument('--query_label', type=str, default=None,
                        help='Query label (or ID of the person). '
                             '**REQUIRED** if flag --operation is extract_features and --image_query is used.')

    # video processing options
    parser.add_argument('--video_path', type=str, default=None, help='Path to the videos')

    # dataset options
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If set, the whole dataset is processed [Options: LFW | LFW_UPDATE | YALEB]')
    parser.add_argument('--specific_dataset_folder_name', type=str, default=None,
                        help='Specific dataset folder. **REQUIRED** if flag --dataset is used.')
    parser.add_argument('--img_extension', type=str, default='jpg',
                        help='Extension of the images [Options: jpg | pgm | txt (in base64 format)]')
    parser.add_argument('--preprocessing_method', type=str, default=None,
                        help='Preprocessing method [Options: None | mtcnn | sphereface | openface]')

    # model options
    parser.add_argument('--model_name', type=str, required=True, default='mobilefacenet',
                        help='Model to test [Options: mobilefacenet | mobiface | sphereface | '
                             'openface | facenet | shufflefacenet | arcface]')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model. If not set, the original trained model will be used.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to extract features')

    args = parser.parse_args()
    print(args)

    # selecting the size of the crop based on the network
    if args.model_name == 'mobilefacenet' or args.model_name == 'sphereface':
        crop_size = (96, 112)
    elif args.model_name == 'mobiface' or args.model_name == 'shufflefacenet' or args.model_name == 'arcface' or args.model_name == 'curricularface':
        crop_size = (112, 112)
    elif args.model_name == 'openface':
        crop_size = (96, 96)
    elif args.model_name == 'facenet':
        crop_size = (160, 160)
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")

    if args.dataset is not None:
        # process whole dataset
        assert args.specific_dataset_folder_name is not None, 'To process a dataset, ' \
                                                              'the flag --specific_dataset_folder_name is required.'
        process_dataset(args.operation, args.model_name, args.batch_size,
                        args.dataset, args.specific_dataset_folder_name,
                        args.img_extension, args.preprocessing_method, crop_size,
                        args.result_sample_path, args.feature_file, args.gpu)
    elif args.image_query is not None:
        process_image(args.operation, args.model_name, args.model_path, args.image_query, args.query_label,
                      args.preprocessing_method, crop_size, args.feature_file, args.gpu)
    elif args.video_path is not None:
        process_video(args.video_path, args.feature_file, save_dir=None,
                      model=load_net(args.model_name, args.model_path, gpu=args.gpu))
    else:
        raise NotImplementedError("Dataset OR Image Path flags must be set. Both flags are None.")
