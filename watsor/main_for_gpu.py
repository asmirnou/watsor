import os
import subprocess
from multiprocessing import set_start_method
from argparse import ArgumentParser

if __name__ == '__main__':
    set_start_method('spawn')

    parser = ArgumentParser(description='Object detection for video surveillance')
    parser.add_argument("--model-path",
                        dest='model_path', metavar='MODEL_PATH',
                        default=os.path.join(os.getcwd(), 'model'),
                        help="path to log file")

    args, unknown = parser.parse_known_args()

    for ext in ('onnx', 'uff'):
        if os.path.isfile(os.path.join(args.model_path, 'gpu.{}'.format(ext))) and not \
                os.path.isfile(os.path.join(args.model_path, 'gpu.trt')):
            engine = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine.py')
            subprocess.run(['python3', '-u', engine,
                            '-i', os.path.join(args.model_path, 'gpu.{}'.format(ext)),
                            '-o', os.path.join(args.model_path, 'gpu.trt'),
                            '-p', os.getenv('TRT_FLOAT_PRECISION', '32')
                            ], check=True)
            break

    from watsor.main import Application

    app = Application()
    app.run()
