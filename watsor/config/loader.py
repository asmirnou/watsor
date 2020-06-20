import os
import re
import yaml
import cerberus
import logging
from watsor.config.schema import schema

_LOGGER = logging.getLogger(__name__)

_ENV_pattern = re.compile('.*?\${(\w+)}.*?')

SECRETS_YAML = "secrets.yaml"

__SECRET_CACHE = {}


def _load_yaml(filename: str, loader: yaml.Loader = yaml.SafeLoader):
    _LOGGER.debug("Loading %s", filename)
    with open(filename, encoding="utf-8") as stream:
        return yaml.load(stream, Loader=loader)


def _env_yaml(loader: yaml.Loader, node: yaml.nodes.Node):
    """Extracts the environment variable from the node's value

    :param loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the environment variable
    """

    value = loader.construct_scalar(node)
    match = _ENV_pattern.findall(value)  # to find all env variables in line
    if match:
        full_value = value
        for g in match:
            full_value = full_value.replace(
                f'${{{g}}}', os.environ.get(g, g)
            )
        return full_value
    return value


def _env_var_yaml(loader: yaml.Loader, node: yaml.nodes.Node) -> str:
    """Load environment variables and embed it into the configuration YAML.

    :param loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the environment variable
    """

    args = node.value.split()

    # Check for a default value
    if len(args) > 1:
        return os.getenv(args[0], " ".join(args[1:]))
    if args[0] in os.environ:
        return os.environ[args[0]]
    raise ValueError(node.value)


def _load_secret_yaml(filename: str):
    """Load the secrets yaml from file.

    :param filename: the name of the secret file
    :return: the contents of the secret file as a dictionary
    """

    if filename in __SECRET_CACHE:
        return __SECRET_CACHE[filename]

    try:
        secrets = _load_yaml(filename)
        if secrets is None:
            raise FileNotFoundError()
        if not isinstance(secrets, dict):
            raise ValueError("Secrets is not a dictionary")
    except FileNotFoundError:
        secrets = {}

    __SECRET_CACHE[filename] = secrets
    return secrets


def _secret_yaml(loader: yaml.Loader, node: yaml.nodes.Node):
    """Load secrets and embed it into the configuration YAML.

    :param loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the secret
    """

    secret_path = os.path.dirname(loader.name)
    while True:
        filename = os.path.join(secret_path, SECRETS_YAML)
        secrets = _load_secret_yaml(filename)

        if node.value in secrets:
            _LOGGER.debug("Secret \"%s\" retrieved from %s", node.value, filename)
            return secrets[node.value]

        secret_path = os.path.dirname(secret_path)
        if not os.path.exists(secret_path) or len(secret_path) < 5:
            break  # Somehow we got past the config folder

    raise ValueError(f"Secret \"{node.value}\" not defined")


def parse(filename=None, data=None):
    """
    Load a yaml configuration file and resolve any environment variables and secrets.

    :param str filename: the path to the yaml file
    :param str data: the yaml data of file is not provided
    :return: the dict configuration or None if file is empty
    :rtype: dict[str, object]
    """

    loader: yaml.Loader = yaml.SafeLoader

    loader.add_implicit_resolver('!ENV', _ENV_pattern, None)
    loader.add_constructor('!ENV', _env_yaml)
    loader.add_constructor("!env_var", _env_var_yaml)
    loader.add_constructor("!secret", _secret_yaml)

    __SECRET_CACHE.clear()
    try:
        if filename:
            return _load_yaml(filename, loader=loader)
        elif data:
            return yaml.load(data, Loader=loader)
        else:
            raise ValueError('Either filename or data should be defined as input')
    finally:
        __SECRET_CACHE.clear()


class _ExtendedValidator(cerberus.Validator):
    """Performs project-specific normalization of configuration:
    - copies ffmpeg and detect defaults to the settings of a camera;
    - checks uniqueness of detection labels and camera names in Yaml lists.
    """

    def _normalize_default_setter_ffmpeg(self, _):
        return self.root_document['ffmpeg']

    def _normalize_default_setter_detect(self, _):
        return self.root_document['detect']

    def _validate_uniquekey(self, uniquekey, field, value):
        """Ensure value uniqueness.
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if uniquekey:
            before = [self.document[i] for i in range(0, field)]
            this_key = next(iter(value))
            before_keys = filter(lambda item: item == this_key,
                                 map(lambda doc: next(iter(doc)), before))
            count = sum(1 for _ in before_keys)
            if count > 0:
                msg = "'{}' is already defined".format(this_key)
                self._error(field, msg)


def _walk_dict(a_dict, visitor, path=()):
    """Walks through a dictionary tree (where a leaf can be also a dictionary)
    without recursion.

    :param a_dict: a dictionary tree to walk through
    :param visitor: callable to invoke for each non-dictionary leaf found
    :param path: start path
    """
    stack = [(a_dict, path)]
    while stack:
        dict_node, path_node = stack.pop(0)
        for key, value in dict_node.items():
            next_path_node = path_node + (str(key),)
            for item in value:
                if isinstance(item, dict):
                    stack.append((item, next_path_node))
                else:
                    visitor(next_path_node, item)


def validate(config):
    """
    Validate configuration against embedded schema.

    :param config: configuration object
    :raises: AssertionError if config is None, ValueError if validation failed
    """

    assert config is not None, "Configuration file is empty"
    validator = _ExtendedValidator(schema)

    if not validator.validate(config, schema):
        str_list = ["Invalid configuration:"]
        _walk_dict(validator.errors, lambda paths, error: str_list
                   .append("\t\"{}\": {}"
                           .format(".".join(paths), error)))
        raise ValueError("\n".join(str_list))

    return validator.normalized(config)


def normalize(config, path):
    """Normalizes config inserting input and output parameters in FFmpeg command line.
    Resolves relative paths in mask file.

    :param config: config object
    :param path: path to the file, where config was loaded from
    :return: normalized config object
    """

    for camera in config['cameras']:
        camera_name = next(iter(camera))
        camera_config = camera[camera_name]
        ffmpeg = camera_config['ffmpeg']

        item = 'decoder'
        ffmpeg[item].insert(0, 'ffmpeg')
        input_index = ffmpeg[item].index('-i')
        ffmpeg[item].insert(input_index + 1, camera_config['input'])
        ffmpeg[item].append('-')

        item = 'encoder'
        if item in ffmpeg:
            ffmpeg[item].insert(0, 'ffmpeg')
            input_index = ffmpeg[item].index('-i')
            size = '{}x{}'.format(camera_config['width'], camera_config['height'])
            ffmpeg[item].insert(input_index, '-s')
            ffmpeg[item].insert(input_index + 1, size)
            ffmpeg[item].insert(input_index + 3, '-')
            if 'output' in camera_config:
                ffmpeg[item].append(camera_config['output'])
            else:
                ffmpeg[item].append('-')

        if 'mask' in camera_config:
            mask = camera_config['mask']
            if not os.path.isabs(mask):
                camera_config['mask'] = os.path.realpath(os.path.join(path, mask))

    return config
