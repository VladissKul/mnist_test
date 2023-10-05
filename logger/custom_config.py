from logger.base_config import BaseConfig


class CustomConfig(BaseConfig):
    """
    self.config: dict
    """

    def __init__(self, config_name: str):  # , session_uuid: str = ''):
        # config_name = f"{config_name.split('.')[0]}_{session_uuid}.{config_name.split('.')[-1]}"
        super(CustomConfig, self).__init__(config_name=config_name)

    def load_full_config(self) -> None:
        """
        description
        ----------
        Метод загрузки файла конфигурации целиком
        """
        exec(self.config_loader_dict[self.config_type])

    def save_full_config(self) -> None:
        """
        description
        ----------
        Метод сохранения файла конфигурации целиком
        """
        exec(self.config_saver_dict[self.config_type])

    def load_section_config(self, section: str) -> dict:
        """
        description
        ----------
        Метод загрузки определенной секции файла конфигурации
                
        
        parameters
        ----------
        section: str - загружаемая секция файла конфигурации

        """

        if self.config:
            None
        else:
            exec(self.config_loader_dict[self.config_type])
        return self.config[section]

    def update_value(self, section: str, key: str, value: str, returned: bool = False):
        self.load_full_config()
        self.config[section][key] = value
        self.save_full_config()
        if returned:
            return self.config

    def update_section(self, section_params: dict, returned: bool = False):
        self.load_full_config()
        self.config.update(section_params)
        self.save_full_config()
        if returned:
            return self.config
