import os


def edge_tpus():
    """Yields all available unassigned Edge TPU devices.
    Set CORAL_VISIBLE_DEVICES environmental variable to a comma-separated list of device paths
    to make only those devices visible to the application.
    """

    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        from watsor.detection.edge_tpu import CoralObjectDetector

        env_cvd = os.environ.get("CORAL_VISIBLE_DEVICES")
        visible_devices = [x.strip() for x in env_cvd.split(",")] if env_cvd is not None else []

        devices = list_edge_tpus()
        for idx, device in enumerate(devices):
            device_name = '{}:{}'.format(device['type'], idx)
            if len(visible_devices) > 0 and device_name not in visible_devices:
                continue

            yield device_name, CoralObjectDetector
    except (RuntimeError, ImportError):
        return


def cuda_gpus():
    """Yields all available CUDA GPU devices, if not subject for the following conditions:

    - set CUDA_DEVICE environmental variable to specific device ID to use only the given device.

    - the default device can be specified in the file ~/.cuda_device

    - additionally, you can set CUDA_VISIBLE_DEVICES environmental variable to a comma-separated list
      of device IDs to make only those devices visible to the application.
    """
    ndevices = 0
    try:
        import pycuda.driver as cuda
        from watsor.detection.tensorrt_gpu import TensorRTObjectDetector

        cuda.init()
        ndevices = cuda.Device.count()
    except (RuntimeError, TypeError, ImportError):
        pass
    except cuda.RuntimeError:
        pass
    if ndevices == 0:
        return

    # Is CUDA_DEVICE set?
    device = os.environ.get("CUDA_DEVICE")

    # Is $HOME/.cuda_device set ?
    if device is None:
        try:
            homedir = os.environ.get("HOME")
            assert homedir is not None
            device = (open(os.path.join(homedir, ".cuda_device"))
                      .read().strip())
        except:
            pass

    if device is not None:
        # If either CUDA_DEVICE or $HOME/.cuda_device is set, try to use it
        try:
            device = int(device)
        except Exception as e:
            raise TypeError("CUDA device number (CUDA_DEVICE or ~/.cuda_device)"
                            " must be an integer") from e

        yield device, TensorRTObjectDetector
    else:
        # Otherwise, try to use any available device
        for device in range(ndevices):
            yield device, TensorRTObjectDetector


def cpus():
    """Yields either TensorFlow or TensorFlow Lite cpu-based detector class
    depending on what dependency is installed.
    """
    try:
        from watsor.detection.tensorflow_cpu import TensorFlowObjectDetector
        yield TensorFlowObjectDetector
        return
    except ImportError:
        pass

    try:
        from watsor.detection.tensorflow_lite_cpu import TensorFlowLiteObjectDetector
        yield TensorFlowLiteObjectDetector
        return
    except ImportError:
        pass
