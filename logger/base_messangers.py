import os
import smtplib

from .core_base import Base


class BaseMessangers(Base):

    def __init__(self):
        self.load_config()

    def load_config(self):
        config = {}
        config['sender'] = 'from@testdomain.com'
        config['receivers'] = ['gennadiy.van@rt.ru', 'gennady.van@mail.ru']
        self.config = config

    def send_email(self, email_text):
        smtpObj = smtplib.SMTP('0.0.0.0')
        smtpObj.sendmail(self.config['sender'], self.config['receivers'], self.create_messange(email_text))

    def create_messange(self, email_text):
        message = "\r\n".join((
            f"""Subject:  Error {os.environ['SESSION_UUID']}""",
            "",
            email_text))
        # message = f"""Subject: Error {os.environ['test_run']},
        # {email_text}"""
        print(message)
        return message
