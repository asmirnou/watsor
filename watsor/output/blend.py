import numpy as np
from watsor.filter.mask import get_alpha_channel


class BlendEffect(object):

    def __init__(self, camera_config):
        alpha_channel, _ = get_alpha_channel(camera_config['mask'],
                                             camera_config['width'],
                                             camera_config['height'])
        image_shape = np.append(alpha_channel.shape, 3)

        # Alpha factor
        self.__alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255
        self.__alpha_factor = np.concatenate((self.__alpha_factor,
                                              self.__alpha_factor,
                                              self.__alpha_factor), axis=2)

        # White background image
        self.__white_image = np.ones(image_shape, dtype=np.float32) * 255
        self.__white_image *= (1 - self.__alpha_factor)

        # Blended image template
        self.__blended_image = np.zeros(image_shape, dtype=np.float32)

    def apply(self, image_in, image_out, shape, header_in, header_out):
        np.copyto(self.__blended_image, image_in, casting='safe')

        self.__blended_image *= self.__alpha_factor
        self.__blended_image += self.__white_image

        np.copyto(image_out, self.__blended_image, casting='unsafe')
