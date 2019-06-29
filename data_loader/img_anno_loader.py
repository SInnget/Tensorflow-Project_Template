from pathlib import Path


class ImagePathAnnotationTemplate(object):
    def __init__(self, file_path):
        assert Path(file_path).exists(), f'{file_path} does not exits.'
        with open(file_path) as fp:
            lines = fp.readlines()
        self._files = [line.strip('\n').split(',') for line in lines]
        self._image_paths = (line[0] for line in self._files)
        self._anno_paths = (line[1:] for line in self._files)

    @property
    def data(self):
        return list(self._image_paths), list(self._anno_paths)
