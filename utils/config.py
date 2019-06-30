import json
from bunch import Bunch
import os

class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if self.__immutable__:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(key, value))

        if isinstance(value, dict):
            value = AttrDict(value)

        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def set_immutable(self):
        self.__dict__[AttrDict.IMMUTABLE] = True

        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                self.set_immutable()

        for v in self.values():
            if isinstance(v, AttrDict):
                self.set_immutable()

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("../experiments", config.exp_name,
                                      "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name,
                                         "checkpoint/")
    return config
