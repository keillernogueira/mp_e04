class defaultOpt():
    def __init__(self):
        self.img_size = 640         # Inference size (pixels)
        self.conf_thres = 0.25      # Object confidence threshold
        self.iou_thres = 0.45       # IOU threshold for NMS
        self.max_det = 1000         # Maximum number of detections per image

        self.device = ''            # Cuda device, i.e. 0 or 0,1,2,3 or cpu'

        self.classes = None           # Filter by class: --class 0, or --class 0 2 3

        self.agnostic_nms = False   # Class-agnostic NMS
        self.augment = False        # Augmented inference

        self.line_thickness = 3     # Bounding box thickness (pixels)
        self.hide_labels = False    # Hide labels
        self.hide_conf = False      # Hide confidences

        self.coco_only = False

    def __str__(self):
        return f"{x for x in self.__dict__.items()}"

    def __repr__(self):
        return f"{x for x in self.__dict__.items()}"

class defaultOptTrain():
    def __init__(self):
        self.weights = 'yolov5l.pt'
        self.coco_only = False
        self.cfg = ''
        self.data = 'dataset.yaml'
        self.hyp = 'hyp.scratch.yaml'
        self.epochs = 300
        self.batch_size = 16
        self.img_size = [640, 640]
        self.resume = False
        self.nosave = False
        self.notest = False
        self.cache_images = False
        self.device = ''
        self.multi_scale = False
        self.single_cls = False
        self.sync_bn = False
        self.local_rank = -1
        self.workers = 8
        self.output_path = 'runs/train/exp'
        self.save_period = -1

    def __str__(self):
        return f"{y for y in self.__dict__.items()}"

    def __repr__(self):
        return f"{y for y in self.__dict__.items()}"