import argparse
import json

import commentjson
from tensorflow.python.client import device_lib


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args





class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def load_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


def load_commentjson(json_path):
    with open(json_path, 'r') as fp:
        return commentjson.loads(fp.read())


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
