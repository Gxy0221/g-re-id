import os.path as osp
import random
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VRU(ImageDataset):

    dataset_dir = "VRU"
    dataset_name = "VRU"

    def __init__(self, root='datasets', test_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'Pic')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')

        if test_list:
            self.test_list = test_list
        else:
            self.test_list = osp.join(self.dataset_dir, 'train_test_split/test.txt')
        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(VRU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines() 

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split(' ')[1])
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, f"{imgid}.jpg")
            imgid = int(imgid)
            if is_train:
                vid = f"{self.dataset_name}_{vid}"
                imgid = f"{self.dataset_name}_{imgid}"
            dataset.append((img_path, vid, imgid))

        if is_train: return dataset
        else:
            random.shuffle(dataset)
            vid_container = set()
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery


@DATASET_REGISTRY.register()
class SmallVRU(VRU):
    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_1200.txt')

        super(SmallVRU, self).__init__(root, self.test_list, **kwargs)


@DATASET_REGISTRY.register()
class MediumVRU(VRU):   
    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_2400.txt')

        super(MediumVRU, self).__init__(root, self.test_list, **kwargs)


@DATASET_REGISTRY.register()
class LargeVRU(VRU):
    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_8000.txt')

        super(LargeVRU, self).__init__(root, self.test_list, **kwargs)