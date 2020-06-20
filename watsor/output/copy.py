import numpy as np
from ctypes import memmove, addressof, sizeof


class CopyHeaderEffect(object):

    @staticmethod
    def apply(image_in, image_out, shape, header_in, header_out):
        memmove(addressof(header_out.get_obj()), addressof(header_in.get_obj()), sizeof(header_in.get_obj()))


class CopyImageEffect(object):

    @staticmethod
    def apply(image_in, image_out, shape, header_in, header_out):
        np.copyto(image_out, image_in)
