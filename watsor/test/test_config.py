import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest import TestCase, main
from test.support import EnvironmentVarGuard
from uuid import uuid4
from watsor.config.loader import parse, validate, normalize, SECRETS_YAML


class TestConfig(TestCase):

    def test_empty_config(self):
        empty_yaml = """
        """

        with self.assertRaises(ValueError):
            validate(parse())
        with self.assertRaises(FileNotFoundError):
            validate(parse(filename=str(uuid4())))
        with self.assertRaises(AssertionError):
            validate(parse(data=empty_yaml))

    def test_required(self):
        minimal_config = """
        cameras:
            - porch:
        """

        with self.assertRaises(ValueError) as error:
            validate(parse(data=minimal_config))
        exception = error.exception

        self.assertRegex(exception.args[0], "cameras.0.porch.detect.*empty values not allowed")
        self.assertRegex(exception.args[0], "cameras.0.porch.height.*required field")
        self.assertRegex(exception.args[0], "cameras.0.porch.width.*required field")
        self.assertRegex(exception.args[0], "cameras.0.porch.input.*required field")
        self.assertRegex(exception.args[0], "cameras.0.porch.ffmpeg.decoder.*required field")

    def test_defaults(self):
        minimal_config = """
        ffmpeg:
            decoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
            encoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
        detect:
            - person:
        cameras:
            - porch:
                width: 640 
                height: 480
                input: http://192.168.42.129:8080/video
                mask: porch.png
                detect:
                    - person:
                    - truck: 
                        area: 80
                        confidence: 70           
        """

        config = normalize(validate(parse(data=minimal_config)), os.path.dirname(__file__))

        self.assertEqual(1, len(config['cameras']))
        camera = next(iter(config['cameras']))
        self.assertTrue('porch' in camera)
        self.assertEqual(640, camera['porch']['width'])
        self.assertEqual(480, camera['porch']['height'])
        self.assertEqual(8, len(camera['porch']['ffmpeg']['decoder']))
        self.assertEqual(10, len(camera['porch']['ffmpeg']['encoder']))
        detect = iter(camera['porch']['detect'])
        person = next(detect)
        self.assertTrue('person' in person)
        self.assertEqual(10, person['person']['area'])
        self.assertEqual(50, person['person']['confidence'])
        truck = next(detect)
        self.assertTrue('truck' in truck)
        self.assertEqual(80, truck['truck']['area'])
        self.assertEqual(70, truck['truck']['confidence'])

    def test_unique(self):
        minimal_config = """
        ffmpeg:
            decoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
            encoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
        detect:
            - person:
            - person:
        cameras:
            - porch:
                width: 640 
                height: 480
                input: http://192.168.42.129:8080/video
                detect:
                    - truck:
                    - truck:
            - porch:
                width: 640 
                height: 480
                input: http://192.168.42.129:8080/video
        """
        with self.assertRaises(ValueError) as error:
            validate(parse(data=minimal_config))
        exception = error.exception

        self.assertRegex(exception.args[0], "detect.1.*.'person' is already defined")
        self.assertRegex(exception.args[0], "cameras.1.*.'porch' is already defined")
        self.assertRegex(exception.args[0], "cameras.0.porch.detect.1.*'truck' is already defined")
        self.assertRegex(exception.args[0], "cameras.1.porch.detect.1.*'person' is already defined")

    def test_secrets(self):
        minimal_config = """
        mqtt:
            host: localhost
            username: !secret mqtt_username
            password: !secret mqtt_password
        ffmpeg:
            decoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
            encoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
        detect:
            - person:
        cameras:
            - porch:
                width: 640 
                height: 480
                input: http://192.168.42.129:8080/video
        """

        secrets = """
        mqtt_username: "john"
        mqtt_password: "qwerty"
        """

        with TemporaryDirectory() as tmp_dir:
            tmp_config_file = NamedTemporaryFile(mode='w', encoding="utf-8",
                                                 prefix=__name__, suffix=".yaml",
                                                 dir=tmp_dir, delete=False)
            try:
                tmp_config_file.write(minimal_config)
                tmp_config_file.flush()

                with open(os.path.join(tmp_dir, SECRETS_YAML),
                          'w', encoding="utf-8") as tmp_secrets_file:
                    tmp_secrets_file.write(secrets)
                    tmp_secrets_file.flush()

                    config = validate(parse(filename=tmp_config_file.name))
            finally:
                # Delete explicitly for compatibility with Windows NT
                tmp_config_file.close()
                os.unlink(tmp_config_file.name)

            self.assertEqual("john", config['mqtt']['username'])
            self.assertEqual("qwerty", config['mqtt']['password'])

    def test_env_vars(self):
        minimal_config = """
        http:
            port: 8080
            username: !env_var "ADMIN_USERNAME john"
            password: !env_var "ADMIN_PASSWORD qwerty"
        mqtt:
            host: localhost
            username: !ENV "${MQTT_ACCOUNT}@${MQTT_DOMAIN}"
            password: !ENV "${MQTT_PASSWORD}"
        cameras:
            - porch:
                width: 640 
                height: 480
                input: http://192.168.42.129:8080/video
                ffmpeg:
                    decoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
                    encoder: ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24']
                detect:
                    - person:
        """

        with EnvironmentVarGuard() as env:
            env.set("ADMIN_USERNAME", "admin")
            env.set("ADMIN_PASSWORD", "12345678")
            env.set("MQTT_ACCOUNT", "admin")
            env.set("MQTT_DOMAIN", "example.com")
            env.set("MQTT_PASSWORD", "qwerty")

            config = validate(parse(data=minimal_config))
            self.assertEqual("admin", config['http']['username'])
            self.assertEqual("12345678", config['http']['password'])
            self.assertEqual("admin@example.com", config['mqtt']['username'])
            self.assertEqual("qwerty", config['mqtt']['password'])


if __name__ == '__main__':
    main(verbosity=2)
