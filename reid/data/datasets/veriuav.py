import glob
import os.path as osp
import re
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VeRiUAV(ImageDataset):

    dataset_dir = "/home/gxy/guo/UAV-reid/datasets/VeRi-UAV"
    dataset_name = "VeRi-UAV"

    def __init__(self, root='/home/gxy/guo/UAV-reid/reid/data/datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(VeRiUAV, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        pattern = re.compile(r'([\d]+)_(\d\d)')
        
        data = []
        for img_path in img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups())         
            if pid == -1: continue  
            assert 0 <= camid <= 7
            camid -= 1
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        return data
