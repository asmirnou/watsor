from watsor.config.coco import COCO_CLASSES

schema = {
    'http': {
        'type': 'dict',
        'default': {},
        'schema': {
            'port': {
                'type': 'integer',
                'min': 1,
                'max': 65535,
                'default': 8080
            },
            'username': {
                'type': 'string',
                'nullable': False
            },
            'password': {
                'type': 'string',
                'nullable': False,
                'dependencies': 'username'
            }
        }
    },
    'mqtt': {
        'type': 'dict',
        'schema': {
            'host': {
                'type': 'string',
                'nullable': False,
                'required': True
            },
            'port': {
                'type': 'integer',
                'min': 1,
                'max': 65535,
                'default': 1883
            },
            'username': {
                'type': 'string',
                'nullable': False
            },
            'password': {
                'type': 'string',
                'nullable': False,
                'dependencies': 'username'
            }
        }
    },
    'ffmpeg': {
        'type': 'dict',
        'default': {},
        'schema': {
            'decoder': {
                'type': 'list',
                'default': [],
                'schema': {
                    'type': 'string',
                    'coerce': str
                }
            },
            'encoder': {
                'type': 'list',
                'schema': {
                    'type': 'string',
                    'coerce': str
                }
            }
        }
    },
    'detect': {
        'type': 'list',
        'default': [],
        'schema': {
            'type': 'dict',
            'maxlength': 1,
            'uniquekey': True,
            'keysrules': {
                'type': 'string',
                'coerce': str,
                'empty': False
            },
            'valuesrules': {
                'type': 'dict',
                'default': {},
                'schema': {
                    'area': {
                        'type': 'float',
                        'min': 0,
                        'max': 100,
                        'default': 10
                    },
                    'confidence': {
                        'type': 'float',
                        'min': 0,
                        'max': 100,
                        'default': 50
                    },
                    'zones': {
                        'type': 'list',
                        'default': [],
                        'schema': {
                            'type': 'integer'
                        }
                    }
                }
            }
        }
    },
    'cameras': {
        'type': 'list',
        'required': True,
        'empty': False,
        'schema': {
            'type': 'dict',
            'maxlength': 1,
            'uniquekey': True,
            'keysrules': {
                'type': 'string',
                'coerce': str,
                'empty': False,
            },
            'valuesrules': {
                'type': 'dict',
                'default': {},
                'schema': {
                    'width': {
                        'type': 'integer',
                        'required': True,
                        'min': 1
                    },
                    'height': {
                        'type': 'integer',
                        'required': True,
                        'min': 1
                    },
                    'input': {
                        'type': 'string',
                        'nullable': False,
                        'required': True,
                        'coerce': str
                    },
                    'output': {
                        'type': 'string',
                        'nullable': False,
                        'dependencies': 'ffmpeg.encoder',
                        'coerce': str
                    },
                    'mask': {
                        'type': 'string',
                        'nullable': False,
                        'coerce': str
                    },
                    'ffmpeg': {
                        'type': 'dict',
                        'default_setter': 'ffmpeg',
                        'schema': {
                            'decoder': {
                                'type': 'list',
                                'required': True,
                                'contains': ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24'],
                                'schema': {
                                    'type': 'string',
                                    'coerce': str
                                }
                            },
                            'encoder': {
                                'type': 'list',
                                'required': False,
                                'contains': ['-i', '-f', 'rawvideo', '-pix_fmt', 'rgb24'],
                                'schema': {
                                    'type': 'string',
                                    'coerce': str
                                }
                            }
                        }
                    },
                    'detect': {
                        'type': 'list',
                        'default_setter': 'detect',
                        'required': True,
                        'empty': False,
                        'schema': {
                            'type': 'dict',
                            'maxlength': 1,
                            'uniquekey': True,
                            'keysrules': {
                                'type': 'string',
                                'coerce': str,
                                'allowed': COCO_CLASSES,
                                'empty': False
                            },
                            'valuesrules': {
                                'type': 'dict',
                                'default': {},
                                'schema': {
                                    'area': {
                                        'type': 'float',
                                        'min': 0,
                                        'max': 100,
                                        'default': 10
                                    },
                                    'confidence': {
                                        'type': 'float',
                                        'min': 0,
                                        'max': 100,
                                        'default': 50
                                    },
                                    'zones': {
                                        'type': 'list',
                                        'default': [],
                                        'schema': {
                                            'type': 'integer'
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
