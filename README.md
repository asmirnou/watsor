# Watsor

Watsor detects objects in video stream using deep learning-based approach. Intended primarily for surveillance it works in sheer real-time analysing the most recent frame to deliver fastest reaction against a detected threat.

## Table of contents
  * [What it does](#what-it-does)
  * [Getting started](#getting-started)
  * [Configuration](#configuration)
    + [Cameras](#cameras)
    + [FFmpeg decoder and encoder](#ffmpeg-decoder-and-encoder)
    + [Detection classes and filters](#detection-classes-and-filters)
    + [Zones and masks](#zones-and-masks)
    + [Tips](#tips)
    + [Environmental variables](#environmental-variables)
    + [Secrets](#secrets)
    + [HomeAssistant integration](#homeassistant-integration)
  * [Running Watsor](#running-watsor)
    + [Docker](#docker)
    + [Python module](#python-module)
      - [Object detection models](#object-detection-models)
      - [Hardware acceleration drivers](#hardware-acceleration-drivers)
  * [Building from source](#building-from-source)
  * [Troubleshooting](#troubleshooting)
  * [Credits](#credits)
  * [License](#license)

## What it does

- Performs smart detection based on artificial neural networks significantly reducing false positives in video surveillance.
- Capable to limit detection [zones](#zones-and-masks) using [mask image](#zones-and-masks) with alpha channel.
- Supports multiple hardware accelerators such as [The Coral USB Accelerator](https://coral.ai/products/accelerator/) and [Nvidia CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to speed up detection algorithms. 
- Reports the detected objects via [MQTT](http://mqtt.org/) protocol primarily for integration with [HomeAssistant](https://www.home-assistant.io/).
- Allows to control video decoder using the commands published over MQTT.
- Broadcasts video stream with rendered object detections in [MPEG-TS](https://en.wikipedia.org/wiki/MPEG_transport_stream) and [Motion JPEG](https://en.wikipedia.org/wiki/Motion_JPEG) formats over HTTP.
- Captures video from any source and encodes video with rendered object detections in any format supported by [FFmpeg](https://ffmpeg.org/).

Being applicable in CCTV, Watsor also suits other areas, where object detection in video stream is required.    

## Getting started

Watsor is written on Python 3 and ships mainly as [Python module](#python-module). The easiest way, however, is trying it in [Docker](#docker) as far as hardware drivers and models are bundled in the images.

Regardless of how you run Watsor, you need to prepare configuration file, describing the cameras you've got and few other options such as types of detections and zones. Refer to the guide below.


## Configuration

Watsor uses the [YAML](https://yaml.org/) syntax for configuration. The basics of YAML syntax are block collections and mappings containing key-value pairs. Each item in a collection starts with a `-` while mappings have the format `key: value`. Watsor performs config validation and, if you specified duplicate keys, missed some values or have wrong YAML syntax, prints the error message at the start. 

Please find an example [configuration](config/config.yaml) file with explanatory comments that you can use as a template.

### Cameras

`cameras` block in document root is a list of cameras you gonna use for surveillance. Each camera must have unique name distinguishing it from others. The name of the camera appears in logs and also is included in path to a video stream over HTTP.

<details>
<summary>Expand code snippet</summary>

```yaml
cameras:
  - camera1:
      width: 640
      height: 480
      input: rtsp://192.168.0.10:554
      #output: !ENV "${HOME}/Videos/camera1.m3u8"
      #mask: camera1.png
      #detect: []
      #ffmpeg: []
  - ...
```
</details>

A camera must specify `width`, `height` and `input` source of a video feed. Refer to the camera setting to figure out the proper values. `width` and `height` are supposed to reflect the resolution of the actual video feed. They don’t change the size of the video, but are used to configure the frame buffers and other things needed for Watsor to work. If `width` or `height` values do not match video feed, the image can’t be displayed correctly. The input source URL usually starts with `rtsp://` or `http://`, but can be anything FFmpeg can consume, a `.mp4` file for instance.

Optional `mask` is a filename of a mask image used to limit detection zone. Refer to the [following](#zones-and-masks) section.

Optional `detect` block overrides [the defaults](#detection-classes-and-filters) to specify the types of objects of interest and set of filters to sift less significant detections.

Optional `ffmpeg` block overrides [the defaults](#ffmpeg-decoder-and-encoder) to specify the arguments for FFmpeg application used to decode input video stream and also to encode the output when necessary. `ffmpeg.encoder` is absolutely optional, define it only if you'd like to record or broadcast video stream with rendered object detections.

When `ffmpeg.encoder` is used to record video file, use `output` key to specify the location. Refer to [FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html) for all available options.  When `output` key is absent, Watsor runs lightweight HTTP server for streaming the video over the network. The streaming is performed in [MPEG-TS](https://en.wikipedia.org/wiki/MPEG_transport_stream) format, so make sure [mpegts format](https://ffmpeg.org/ffmpeg-formats.html#mpegts) is set in `ffmpeg.encoder` output arguments. When `output` is set the encoded stream is being written to file and the broadcast in MPEG-TS over HTTP is not possible.

Regardless of whether or not `ffmpeg.encoder` is enabled and `output` is defined, the broadcast of video stream in [Motion JPEG](https://en.wikipedia.org/wiki/Motion_JPEG) format over HTTP is available all the time. 

### FFmpeg decoder and encoder

`ffmpeg` block present in document root is used as defaults for decoding and encoding a video stream if a camera does not set its own.

[FFmpeg](https://ffmpeg.org/ffmpeg.html#Description) is a very fast video converter that grabs a live video from source. Watsor runs one or two FFmpeg subprocesses for each camera: mandatory decoder and optional encoder. The decoder reads from the source specified by the camera's `input` key (which can be an URL of camera feed, a regular file or a grabbing device), following the `-i` option. The encoder writes to the output file, if camera's `output` is set, or _stdout_, if not. The options in the command line before and after `-i` determine the parameters for input and output of each FFmpeg subprocess. The input parameters, for example, can enable hardware accelerated decoding. The output of decoder and the input of encoder must include `-f rawvideo -pix_fmt rgb24` as Watsor reads raw video frames (24-bit) from first FFmpeg's _stdout_ and after detection complete can write raw video frames in `stdin` of another FFmpeg subprocess. 

You must include `-i` option in the command line list **not followed** by the input as Watsor includes camera's `input` key after `-i` automatically. For encoder it also includes other required options such as `-s` to define explicitly the size of raw video frames. 

<details>
<summary>Expand code snippet</summary>

```yaml
ffmpeg:
  decoder:
    - ...
    - -i
    - -f
    -  rawvideo
    - -pix_fmt
    -  rgb24
  encoder:
    - -f
    -  rawvideo
    - -pix_fmt
    -  rgb24
    - -i 
    - -f
    -  mpegts
    - ... 
```
</details>

`ffmpeg.encoder` is optional and intended for the recording or broadcasting of video stream with rendered object detections. The recording is suitable for on demand streaming as the video is stored in a file such as `.m3u8` (HLS) or `.mp4` and can be re-watched. The broadcasting means the video is being recorded in real time and the viewer can only watch it as it is streaming. To broadcast the video with rendered object detections over HTTP set `-f mpegts` option in the encoder and remove the camera's `output` key. The link to the video stream can be grabbed from Watsor's home page and opened in media player such as [VLC](https://en.wikipedia.org/wiki/VLC_media_player).  

When broadcasting live video stream in MPEG-TS be aware of noticeable [latency](https://trac.ffmpeg.org/wiki/StreamingGuide#Latency), which is unavoidable due to the nature of video encoding algorithm. Bear in mind the media player introduces a latency of its own as it buffers the feed. To watch the video with rendered object detections in sheer real-time with zero latency open Motion JPEG URL of the camera feed.

Broadcasting in MPEG-TS format is perfect for streaming over the network as the video sequencing compression transports only the changes in the sequence thus uses less network bandwidth (and storage). Motion JPEG  compresses each video frame into a JPEG image making them available as a continuous flow of images over a network. As all frames are compressed independently and entirely, the amount of data transmitted across the network is far more than that of MPEG-TS. 

### Detection classes and filters

`detect` block present in document root is used as defaults if a camera does not set its own. The block is a list of [COCO classes](https://github.com/asmirnou/watsor/blob/master/watsor/config/coco.py#L16-L105) (90 at max) used for detection each specifying three filters: `area`, `confidence` and `zones`.      

<details>
<summary>Expand code snippet</summary>

```yaml
detect:
  - person:
      area: 20                   
      confidence: 60              
      zones: [1, 3, 5]   
  - car:
  - ...          
```
</details>

- The `area` is the minimum area of the bounding box an object should have in order to be detected. Defaults to 10% of entire video resolution if not set.
- `confidence` is a minimum threshold that a detection is what it's guessed to be, otherwise it's ruled out. 50% if not set.
- `zones` is a list of indexes designating the zones on mask image, where detection is allowed. Not set by default meaning all zones are allowed.

### Zones and masks

To limit detection zones **take a snapshot** from your camera. The width and height of the image must be of the same size as your video feed. 

Open the image in graphics editor such as GNU Image Manipulation Program (GIMP) and select the desired zone. Make the selection floating, then create a new layer from the given floating selection. That layer represents a zone, outside of which the objects are not being detected. 

Select main layer with the rest of the image and change the **opacity** of the layer to `~85%`. The opacity of the layers from floating selection must remain `100%` to let Watsor distinct them from the background. Save image in PNG 32-bit format with alpha channel. Specify the file name and path (can be relative) in camera configuration.

The following depicts said above:

![Zones demo](config/zones_demo.jpeg) 

You can select many zones and of any shape, but make sure they don't overlap, otherwise several zones merge in one.

When the **bounding box** of an object intersects with the shape of a zone, detection occurs and the zone highlights in yellow. The index of a zone is then transmitted over MQTT. The zones are indexed in mask image depending on the proximity of their center to the origin. On the sample above there are two zones and a car was detected in `2` (shown in red). If not very clear of what zone is the index, the following tool will display it for you:   

```bash
python3 -m watsor.zones -m config/porch.png
```

### Tips
- Not using [hardware accelerator](#hardware-acceleration-drivers) results in high CPU load. In order to reduce CPU load change the **frame rate** of an input video stream either in the setings of your camera or using FFmpeg [filter](https://trac.ffmpeg.org/wiki/ChangingFrameRate) as follows:

    <details>
    <summary>FPS filter</summary>
     
    ```yaml
    - -filter:v
    -  fps=fps=10
    ``` 
    </details>
    
    Ideally, the input frame rate of the cameras should not exceed the capabilities of the detectors, otherwise all available CPU units will be used for detection.

- Unless you record video with rendered object detections choose **smaller resolution** in camera settings. 300x300 is the size of the images used to train the object detection model. During detection the framework automatically converts an input image to the size of the model to match the size of its tensors. As the resize happens anyway during the detection, feeding the stream of high resolution doesn’t make much sense unless the output stream the Watsor produces (the size of which matches the input resolution) is being recorded for later review (where the high resolution video is obviously desired).
    
    Sometimes camera doesn't provide much options to change resolution. Use FFmpeg [scale filter](https://trac.ffmpeg.org/wiki/Scaling) then:
    
    <details>
    <summary>scale filter</summary>
    
    ```yaml
    - -filter:v
    -  scale=640:480
    ```
    </details>
    
    If you need to combine scale filter with FPS filter, separate them with [commas](https://trac.ffmpeg.org/wiki/FilteringGuide):
    
    <details>
    <summary>two filters together</summary>
    
    ```yaml
    - -filter:v
    -  fps=fps=10,scale=640:480
    ```
    </details>

- Consider configuring **hardware acceleration** for decoding H.264 video (or whatever your camera produces) in FFmpeg. The command line options with `hwaccel` prefix are right for that. Refer to the following [wiki](https://trac.ffmpeg.org/wiki/HWAccelIntro) to learn what methods are available on your device.

- Playing the video in VLC (or other player) require constant rate of 25 frames / sec. Having many cameras or very few detectors processing all video sources sometimes can not provide such an output speed. Let's say you've got 2 cameras each producing 30 FPS and only one CPU-based detector capable to cope only with 25 FPS. The output speed of MPEG-TS video stream then will be only 12.5 FPS for each camera, resulting in jerks and pauses while viewing a video. To make a video fluent, the frame rate has to be changed, such that the output frame rate is higher than the input frame rate. 
    
    <details>
    <summary>The following trick may be used (expand code snippet)</summary>
    
    ```yaml
    encoder:
      - -hide_banner
      - -f
      -  rawvideo
      - -pix_fmt
      -  rgb24
      - -r
      -  10
      - -vsync
      -  drop
      - -i
      - -an
      - -f
      -  mpegts
      - -r
      -  30000/1001
      - -vsync
      -  cfr
      - -vcodec
      -  libx264
      - -pix_fmt
      -  yuv420p
    ```
    </details>
    
    First the rate is lowered even more down to 10 FPS in order to guarantee constant feed for FFmpeg encoder (`-r 10`). The frames exceeding 10 FPS are dropped (`-vsync drop`) in order to match the target rate. Then output speed is set to be standard `30000/1001` (~30 FPS) and constant (`-vsync cfr`) to produce fluent video stream, duplicating frames as necessary to achieve the targeted output frame rate.

### Environmental variables
 
You can include values from your system’s environment variables either like [Home Assistant Core does](https://www.home-assistant.io/docs/configuration/yaml/#environmental-variables):
 
 `password: !env_var PASSWORD default_password` or 
 
 `input: !ENV "rtsp://${RTSP_USERNAME}:${RTSP_PASSWORD}@192.168.0.10:554`

### Secrets

You can remove any private information from your configuration file. The entries for password can be replaced with `!secret` keyword followed by an identifier. The password will be then looked at `secrets.yaml` file  containing the corresponding password assigned to the identifier:

<details>
<summary>config.yaml</summary>

```yaml
mqtt:
  username: !secret mqtt_username
  password: !secret mqtt_password
``` 
</details>

<details>
<summary>secrets.yaml</summary>

```yaml
mqtt_username: "john"
mqtt_password: "qwerty"
```
</details>

A `secrets.yaml` will be resolved first by looking in the same folder as the YAML file referencing the secret,  next, parent folders will be searched for a `secrets.yaml` file. The logic inherits from Home Assistant and you can place your config along with HA files reusing single `secrets.yaml` file for both apps.

### HomeAssistant integration

Watsor can communicate with [HomeAssistant](https://www.home-assistant.io/) via [MQTT](http://mqtt.org/) - "Internet of Things" connectivity protocol. Please refer to [demo project](https://github.com/asmirnou/watsor-haas) to examine the configuration files.

<details>
<summary>To configure optional MQTT client add the following lines (expand the snippet):</summary>

```yaml
mqtt:
  host: localhost
  #port: 1883
  #username: !secret mqtt_username
  #password: !secret mqtt_password  
```
</details>

List of MQTT topics:
- `watsor/cameras/{camera}/available`
- `watsor/cameras/{camera}/command`
- `watsor/cameras/{camera}/sensor`
- `watsor/cameras/{camera}/state`
- `watsor/cameras/{camera}/detection/{class}/state`
- `watsor/cameras/{camera}/detection/{class}/details`

The binary state (`ON` / `OFF`) of a detected object class is published at `/detection/{class}/state` topic, confirming the state every 10 seconds. This signals about a detected threat and is supposed to trigger an alarm.

Subscribe to the topic `available` to receive availability (online/offline) updates about specific camera. The camera is `online` when Watsor starts and goes `offline` when the application stops.

The camera can be controlled via topic `command` allowing to start/stop the decoder, limit frame rate and enable/disable the reporting of detection details.
- When alarm control panel is armed, send `ON` to start the decoder and detection processes. When disarmed - send `OFF` to stop analysing the camera feed. The camera notifies about its running state through the `state` topic.
- If nothing is detected for more than 30 seconds, you can slow down the camera by sending `FPS = 5` (or whatever rate you prefer) to limit frame rate in order to reduce the load on CPU or hardware accelerated detector. The camera will reset FPS limit itself and will reach full speed as soon as something's detected. 
- The reporting of detection details including confidence, coordinates of the bounding box and zone index along with the timestamp of the frame can be turned on / off by sending `details = on` / `details = off` accordingly. The details are published at another topic in JSON format as follows:
    ```json
    {
        "t": "2020-06-27T17:41:21.681899",
        "d": [{
            "c": 73.7,
            "b": [54, 244, 617, 479]
        }]
    }
    ```
    where `t` is timestamp of a frame, `d` - array of all detections of the given class of an object designated in topic path. Each detection has confidence `c` and bounding box `b` consisting of `x_min`, `y_min`, `x_max`, `y_max`. 

The topic `sensor` is used to send the updates about camera current input and output speeds. Monitoring the speed is useful to trigger the alert, if camera is broken or disconnected suddenly.

The sensor values and detection details can be transmitted over MQTT very often up to tens times per second. The [recorder](https://www.home-assistant.io/integrations/recorder/) integration in HomeAssistant constantly saves data, storing everything by default from sensors to state changes. Fortunately, HomeAssistant allows to customize what needs to be written and not. A good idea is to include in recoder only those measurements that are really needed to avoid degradation of HomeAssistant's performance.

## Running Watsor

### Docker

To run Watsor in Docker mount [configuration](#configuration) files at `/etc/watsor` folder of a container. Add host devices for video and detection hardware acceleration (if available). As far as Watsor runs several processes, which share the memory, increase the default shared memory size of 64m to a bit more depending on the number of cameras. Here is example of [docker-compose.yaml](docker/docker-compose.yaml):

<details>
<summary>Expand code snippet</summary>

```yaml
version: '3'

services:
  watsor:
    container_name: watsor
    image: smirnou/watsor:latest
    environment:
      - LOG_LEVEL=info
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ../config:/etc/watsor:ro
    devices:
      - /dev/bus/usb:/dev/bus/usb
      - /dev/dri:/dev/dri
    ports:
      - 8080:8080
    shm_size: 512m
``` 
</details>

To run GPU accelerated `watsor.gpu` Docker image use the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)). 
Pass `--gpus all` flag to add GPU devices to the container:

<details>
<summary>Expand code snippet</summary>

```bash
docker run -t -i \
    --rm \
    --env LOG_LEVEL=info \
    --volume /etc/localtime:/etc/localtime:ro \
    --volume $PWD/config:/etc/watsor:ro \
    --device /dev/bus/usb:/dev/bus/usb \
    --device /dev/dri:/dev/dri \
    --publish 8080:8080 \
    --shm-size=512m \
    --gpus all \
    smirnou/watsor.gpu:latest
```
</details>

If your GPU supports Half precision (also known as FP16), you can boost performance by enabling this mode as follows:

`docker run --gpus all --env TRT_FLOAT_PRECISION=16 ...` 

Please note that native GPU support has not landed in docker-compose [yet](https://github.com/docker/compose/issues/6691).

Models for CPU/GPU and EdgeTPU (Coral) are bundled in Docker images. You can use your own models, trained based on those listed in [object detection models](#object-detection-models) section, by mounting the volume at `/usr/share/watsor/model`.

The following table lists the available docker images:

| Image | Suitable for |
|---|---|
| [watsor](https://hub.docker.com/r/smirnou/watsor) | x86-64 |
| [watsor.gpu](https://hub.docker.com/r/smirnou/watsor.gpu) | x86-64 with Nvidia CUDA GPU  |
| [watsor.pi4](https://hub.docker.com/r/smirnou/watsor.pi4) | Raspberry PI 4 with 64-bit OS |  

### Python module

Watsor works well on Pyhton 3.6, 3.7, 3.8. Use a [virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) when installing Python packages.

1. Install module:

   ```bash
   python3 -m pip install[cpu] watsor
   ```
   
   If you've got a [hardware accelerator](#hardware-acceleration-drivers) or are installing the application on a tiny board like Raspberry Pi, take a look at _extra_ profiles `coral`, `cuda` or `lite` in [setup.py](setup.py). The dependencies listed in those profiles need to be installed in advance. Refer to the documentation of the device or take a look at one of the [Docker files](docker/) to understand how the dependencies are installed. 

2. Create `model/` folder, download, unpack and prepare the [object detection models](#object-detection-models) (see the section below).

3. Write the config file by following the [guide](#configuration) above, take [this config](config/config.yaml) as an example.

4. Make sure `ffmpeg` is installed on your operating system.

5. Run watsor as follows:

   ```bash
   python3 -m watsor.main --config config/config.yaml --model-path model/
   ```
   or if NVIDIA CUDA GPU is present and [properly set up](#hardware-acceleration-drivers):
   ```bash
   python3 -m watsor.main_for_gpu --config config/config.yaml --model-path model/
   ```

Open [http://localhost:8080](http://localhost:8080) to navigate to a simple home page, where you'll find the links to the cameras video streams, snapshots of object detected classes and metrics.

<sub>Please note that without the hardware accelerator Watsor uses [TensorFlow](https://www.tensorflow.org/) to run the inference on CPU. Watsor is not bound to specific version of TensorFlow and `pip` likely installs the latest. The object detection API of TensorFlow is not yet converted to `2.2.0` and is still based on `1.5.3`, so you won't get much benefit of the latest TensorFlow, but rather far less disk space. You better pre-install TensorFlow `1.5.3` prior to installing Watsor, unless you use it for other purposes. You also **don't have** to install TensorFlow GPU support as Watsor performs the inference via pure TensorRT when working on GPU.</sub>

#### Object detection models

Watsor uses Single-Shot Detector (SSD) detector trained to recognize 90 classes of object. The detection model has several types providing the trade-off between accuracy and speed. For example, _MobileNet v1_ is the fastest, but less accurate than _Inception v2_.   

The models are available in several formats depending on the device the inference is being performed on. If you've got [The Coral USB Accelerator](https://coral.ai/products/accelerator/) download one of the models built for Edge TPU (MobileNet v1/v2), rename the file and put in `model/` folder as `edgetpu.tflite`. Have [Nvidia CUDA GPU](https://developer.nvidia.com/cuda-gpus) on board - download one of three models, unpack the archive, rename the file and put in `model/` folder as `gpu.uff`.

| Device | Filename | MobileNet v1 | MobileNet v2 | Inception v2 |
|---|---|---|---|---|
| Coral | `model/edgetpu.tflite` | [MobileNet v1](https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite) | [MobileNet v2](https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite) | N/A |
| Nvidia CUDA GPU | `model/gpu.uff` | [MobileNet v1](https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v1.tar.gz) | [MobileNet v2](https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v2.tar.gz) | [Inception v2](https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Inception-v2.tar.gz) |
| CPU | `model/cpu.pb` | [MobileNet v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [MobileNet v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | [Inception v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) |
| Raspberry Pi & others | `model/cpu.tflite` | [MobileNet v1](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) | N/A | N/A | 
  
#### Hardware acceleration drivers

Use of hardware accelerator is optional, but highly recommended as object detection as such requires much computation. Speaking of number of frames per second the CPU of a desktop computer can process just 24 FPS of the lightest model, but an accelerator can boost the performance up to 5 times. Two accelerators connected or installed can handle 200 FPS, which is more than enough for handling several video cameras.

Watsor supports **multiple** accelerators, equally balancing the load among them. Having at least one accelerator connected Watsor uses the hardware for computation, less loading the CPU. It falls over to CPU running Tensorflow if no accelerator is available.

To restrict the devices that Watsor sees set `CORAL_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` environmental variables to a comma-separated list of device paths/IDs to make only those devices visible to the application. The device path/ID can be known from application logs after enabling debug mode with the following command-line option: `--log-level debug`  

If you've got the Coral install the [Edge TPU runtime](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) and [Edge TPU Python library](https://coral.ai/docs/edgetpu/api-intro/#install-the-library-and-examples).

Have [Nvidia CUDA GPU](https://developer.nvidia.com/cuda-gpus) on board - install the following components (now the hard part, you'd better consider [Docker](#docker):
- [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) for your graphic card
- [CUDA® 10.2.89](https://docs.nvidia.com/cuda/archive/10.2/)
- [cuDNN 7.6.5.32](https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_765/cudnn-developer-guide/index.html)
- [TensorRT 6.0.1](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html)
- [PyCUDA 2019.1.2](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html#installing-pycuda)

Please note that having installed GPU support you have to compile also a plugin for UFF model that is not shipped by Watsor Python module. See [building from source](#building-from-source) for reference.

## Building from source

Clone Git repository:

```bash
git clone https://github.com/asmirnou/watsor.git
```

Create virtual environment and install Watsor dependencies:

```bash
make venv

source venv/bin/activate

make install
```

Running Watsor with GPU support requires the [plugin](watsor/plugin/FlattenConcat.cpp). To compile the plugin make sure you have installed CUDA and TensorRT, then run `make plugin` and copy the library either in Python module or system path. 

## Troubleshooting
 
If you encounter issues installing Watsor prerequisites, try to seek for solution over the web first. I'm glad to help, but most of the times the answer is being composed from the findings on https://stackoverflow.com/ and etc. Be bold and [file an issue](https://github.com/asmirnou/watsor/issues) having not found a solution.

## Credits

* Inspired by [Frigate](https://github.com/blakeblackshear/frigate)
* Domain knowledge drawn from [pyimagesearch](https://www.pyimagesearch.com/)

## License

[MIT License](LICENSE)
