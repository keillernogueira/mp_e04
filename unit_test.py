import unittest

from pessoas.train import train
from pessoas.retrieval import retrieval
from pessoas.update_dataset import update_dataset

from yolov5 import train_light
from yolov5 import detect_obj

class Tests(unittest.TestCase):

    def test_person_train(self):
        train('/mnt/DADOS_PONTOISE_1/keiller/mp_e04/pessoas/datasets/CASIA-WebFace/',
              '/mnt/DADOS_PONTOISE_1/keiller/mp_e04/pessoas/outputs/')

    def test_person_retrieval(self):
        retrieval('https://publisher-publish.s3.eu-central-1.amazonaws.com/pb-brasil247/swp/jtjeq9/media/20190830150812_6c55e3b5-c22c-4b41-a48e-3038c5088f31.jpeg',
                  'features.mat', 'outputs/')

    def test_person_update_dataset(self):
        update_dataset("https://upload.wikimedia.org/wikipedia/commons/3/37/Arnold_Schwarzenegger.jpg",
                       img_ID="Arnold_Schwarzenegger", feature_file="features.mat")

    #####################################
    # def test_object_train(self):
    #     # python train_light.py --data datasets/dataset.yaml
    #     train_light.train()
    #
    # def test_object_detect(self):
    #     # python detect_obj.py --weights weights/best.pt --source /mnt/DADOS_PONTOISE_1/keiller/mp_e04/yolov5/datasets/obj_train_data/images/test/DefenseKnifeAttack0103.jpg  --format img --output-path outputs/
    #     detect_obj.retrieval()


if __name__ == '__main__':
    unittest.main()
