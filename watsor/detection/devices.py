import os

try:
    from edgetpu.basic import edgetpu_utils
    from watsor.detection.edge_tpu import CoralObjectDetector
except ImportError:
    pass

try:
    import pycuda.driver as cuda
    from watsor.detection.tensorrt_gpu import TensorRTObjectDetector
except ImportError:
    pass


def edge_tpus():
    """Yields all available unassigned Edge TPU devices.
    Set CORAL_VISIBLE_DEVICES environmental variable to a comma-separated list of device paths
    to make only those devices visible to the application.
    """

    try:
        env_cvd = os.environ.get("CORAL_VISIBLE_DEVICES")
        visible_devices = [x.strip() for x in env_cvd.split(",")] if env_cvd is not None else []

        devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
        for device in devices:
            if len(visible_devices) > 0 and device not in visible_devices:
                continue

            yield device, CoralObjectDetector
    except (RuntimeError, NameError):
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
        cuda.init()
        ndevices = cuda.Device.count()
    except (RuntimeError, TypeError, NameError):
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
