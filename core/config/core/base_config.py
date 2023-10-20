from configparser import ConfigParser

import yaml

from core.config.core.base import Base


class BaseConfig(Base):
    '''
    self.config: dict
    '''

    def __init__(self, config_name: str) -> None:
        self.config_name = config_name
        # self.config_name = os.environ['ConfigName']
        self.init_config_params()
        self.save_config_params()

    def init_config_params(self):
        self.config_type = self.config_name.split('.')[-1]
        self.config_fileways = self.config_name
        self.config_loader_dict = {
            'ini': 'self.load_config_ini()',
            'json': 'self.load_config_json()',
            'yaml': 'self.load_config_yaml()',
            'yml': 'self.load_config_yaml()',
        }

    def save_config_params(self):
        self.config_type = self.config_name.split('.')[-1]
        self.config_saver_dict = {
            'ini': 'self.save_config_ini()',
            'json': 'self.save_config_json()',
            'yaml': 'self.save_config_yaml()',
            'yml': 'self.save_config_yaml()',
        }

    def load_config_yaml(self) -> None:
        with open(self.config_name, encoding='utf-8') as file:
            config = yaml.safe_load(file)
        self.config = config

    def load_config_ini(self) -> None:
        config = ConfigParser(self.config_name)
        config.read()
        self.config = config

    def load_config_json(self) -> None:
        pass

    def save_config_yaml(self) -> None:
        with open(self.config_name, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
