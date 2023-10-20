from core.monitoring.messangers.core.base_messangers import BaseMessangers


class CustomMessangers(BaseMessangers):

    def __init__(self):
        super(CustomMessangers, self).__init__()

    def send_messange(self, email_text):
        self.send_email(email_text)
