import os
import argparse
import time
import json
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from objetos.yolov5.models.experimental import attempt_load
from objetos.yolov5.utils.datasets import LoadStreams, LoadImages
from objetos.yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from objetos.yolov5.utils.plots import colors, plot_one_box
from objetos.yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

from objetos.yolov5.utils.data import read_json, DetectLoadImages
from objetos.yolov5.utils.options import defaultOpt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@torch.no_grad()
def retrieval(img_path, model_path, output_path, save_as, opt=defaultOpt(), output_file='obj_detections.json'):
    """
    - abrir imagem (seja local, ou por download)
    - inferencia usando o modelo salvo
    - gerar a saida de acordo com o formato
    """
    check_requirements(requirements=os.path.join(ROOT_DIR, 'requirements.txt'), exclude=('tensorboard', 'pycocotools', 'thop'))

    assert save_as in ['img', 'json', 'both'], f"Output format <{save_as}> not supported. The available options are: ['img', 'json', 'both']."
    if save_as == 'both': 
        save_as = ['img', 'json']
    else:
        save_as = [save_as]

    if '.json' in img_path: 
        img_path = read_json(img_path)

    imgsz = opt.img_size  # Get Image size to resize inputs if necessary

    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)  # Make output folder directory

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(model_path, map_location=device, coco_only=opt.coco_only)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Dataloader for the input images/videos
    vid_path, vid_writer = None, None

    output_folder = (save_dir / 'dowloaded_files') # Folder for download files if necessary
    dataset = DetectLoadImages(img_path, img_size=imgsz, stride=stride, output_folder=output_folder)

    out_file_data = [] # Used if json output file is generated        

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    for path, img, im0s, vid_cap, hash_data in dataset:
        # Image to tensor
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        print(len(pred))
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()
        print(len(pred))
        # Process detections (for each image in the batch, default 1)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            data_dict = {}
            data_dict['name'] = p.name # Name of the file
            data_dict['path'] = str(p.resolve()) # Absolute path of the file
            data_dict['hash'] = hash_data  # hash of the file
            data_dict['frame'] = str(frame)

            object_id = 1
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if 'json' in save_as:  # Write to file                        
                        object_dict = {}
                        object_dict['id'] = object_id
                        object_dict['class'] = names[int(cls)]
                        object_dict['confidence'] = conf.item()
                        object_dict['box'] = [pxl.item() for pxl in xyxy]

                        data_dict[f'object_{object_id}'] = object_dict
                        
                        object_id += 1

                    if 'img' in save_as:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
            data_dict['objects'] = object_id - 1 

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if 'json' in save_as:
                out_file_data.append(data_dict)

            # Save results (image with detections)
            if 'img' in save_as:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if 'json' in save_as:
        with open (increment_path(save_dir/ output_file) , 'w') as outfile:
            json.dump({'output' : out_file_data}, outfile, indent = 4, separators = (',', ':'))
            s = f"\nOutput file saved in {save_dir} as {outfile.name}" if 'json' in save_as else ''

    print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')
    return out_file_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='data/images',
                        help='Source of images/videos. Can be a folder, a single image/video file, '
                             'a url to a image/video, or a json with a list of files (local or remote)')  # file/folder, 0 for webcam
    parser.add_argument('--model_path', nargs='+', type=str, default='yolov5l.pt', help='model.pt path')
    parser.add_argument('--coco_only', default=False, action='store_true', 
                        help='if the model weigths is not found locally, download the models trained only on coco in the original repository.')

    parser.add_argument('--output_path', type=str, default='outputs/',
                        help='Path where the outputs generetated will be saved')
    parser.add_argument('--output_file', type=str, default='detections.json',
                        help='Name of the output json file (if is created)')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--save_as', default='img', help='output format, options (img|json|both)')
    
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')

    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

    opt = parser.parse_args()
    print(opt)

    retrieval(opt.img_path, opt.model_path, opt.output_path, opt.save_as, opt, output_file=opt.output_file)
