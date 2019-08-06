from .utils import camel_to_snake, InvalidClassNameError



class Factory(object):
    """
    each subclass should have:
    _registered_ = {}
    """

    @classmethod
    def create(cls, name):
        if name in cls._registered_:
            return cls._registered_[name]
        else:
            raise ValueError('{} does not have type {}, only {}'.format(
                cls.__name__, name, cls.list()))

    @classmethod
    def register(cls, registered_cls, suffix=None, ignore=None):
        reg_cls_name = registered_cls.__name__
        if reg_cls_name.startswith('_'):
            return
        if reg_cls_name == ignore:  # ignore base class
            return
        if suffix is not None:
            if reg_cls_name == suffix:  # ignore base class
                return
            if not reg_cls_name.endswith(suffix):
                raise InvalidClassNameError(
                    f'{reg_cls_name} should be end with {suffix}')
            reg_cls_name = reg_cls_name.rsplit(suffix, 1)[0]

        cls._registered_[camel_to_snake(reg_cls_name)] = registered_cls

    @classmethod
    def size(cls):
        return len(cls._registered_)

    @classmethod
    def list(cls):
        return list(cls._registered_.keys())
