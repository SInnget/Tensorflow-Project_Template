from pathlib import Path
from utils.utils import file_reader

class ImagePathAnnotationTemplate(object):
    """ Assume both image and annotation was recorded in text file,
        each line records image path and annoation paths.

    """
    def __init__(self, file_path):
        assert Path(file_path).exists(), f'{file_path} does not exits.'
        self._files = [line.strip('\n').split(',') for line in file_reader(file_path)]
        self._image_paths = [line[0] for line in self._files]
        self._anno_paths = [line[1:] for line in self._files]

    @property
    def data(self):
        return self._image_paths, self._anno_paths
