import copy
import logging
import os
from tabulate import tabulate
from termcolor import colored

logger = logging.getLogger(__name__)

class Dataset(object):
    _junk_pids = []

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self._train = train
        self._query = query
        self._gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose
        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

    @property
    def train(self):
        if callable(self._train):
            self._train = self._train()
        return self._train

    @property
    def query(self):
        if callable(self._query):
            self._query = self._query()
        return self._query

    @property
    def gallery(self):
        if callable(self._gallery):
            self._gallery = self._gallery()
        return self._gallery

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):  
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):        
        pids = set()
        cams = set()
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)

    def get_num_pids(self, data):        
        return self.parse_data(data)[0]

    def get_num_cams(self, data):       
        return self.parse_data(data)[1]

    def show_summary(self):        
        pass

    def combine_all(self):        
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = getattr(self, "dataset_name", "Unknown") + "_test_" + str(pid)
                camid = getattr(self, "dataset_name", "Unknown") + "_test_" + str(camid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self._train = combined

    def check_before_run(self, required_files):       
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


class ImageDataset(Dataset):   
    def show_train(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['train', num_train_pids, len(self.train), num_train_cams]]       
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

    def show_test(self):
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [
            ['query', num_query_pids, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, len(self.gallery), num_gallery_cams],
        ]
        
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
