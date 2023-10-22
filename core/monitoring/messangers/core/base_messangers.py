import smtplib

import yaml

from .base import Base


class BaseMessangers(Base):

    def __init__(self):
        self.load_config()

    def load_config(self):
        with open("config.yml", "r") as config_file:
            config_yml = yaml.safe_load(config_file)
        config = {}
        config['sender'] = config_yml["messangers"]["sender"]
        config['receivers'] = config_yml["messangers"]["receivers"]
        self.config = config

    def send_email(self, email_text):
        with open("config.yml", "r") as config_file:
            config_yml = yaml.safe_load(config_file)
        smtpObj = smtplib.SMTP(config_yml["messangers"]["smtpObj"])
        smtpObj.sendmail(self.config['sender'], self.config['receivers'], self.create_messange(email_text))

    def create_messange(self, email_text):
        message = "\r\n".join((
            f"""Subject:  Error""",
            "",
            email_text))
        # message = f"""Subject: Error {os.environ['test_run']},
        # {email_text}"""
        print(message)
        return message
