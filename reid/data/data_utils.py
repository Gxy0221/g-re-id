import torch
import numpy as np
from PIL import Image, ImageOps
import threading
import queue
from torch.utils.data import DataLoader
from reid.utils.file_io import PathManager

def read_image(file_name, format=None):
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "L":
            image = np.expand_dims(image, -1)
        elif format == "BGR":
            image = image[:, :, ::-1]
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=10):
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():       
            return
        self.iter.exit_event.set()
        for _ in self.iter:
            pass
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(
                        device=self.local_rank, non_blocking=True
                    )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream 
        )
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def shutdown(self):
        self._shutdown_background_thread()
