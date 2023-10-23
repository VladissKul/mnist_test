import os

from core.config.custom_config import CustomConfig


class HdfsStorage:

    def __init__(self):
        pass

    def load_config(self):
        config = CustomConfig(os.environ['ServiceConfigFile'])
        config.load_full_config()
        self.config = config.load_section_config(self.class_name)

    def load_folder(self, hdfs_folderways: str, local_folderways: str):
        """
        description
        ----------
        метод загрузки папки из hdfs в локальное хранилище


        parameters
        ----------
        hdfs_folderways: str, путь до hdfs хранилища
        local_folderways: str, путь до локального хранилища
        """
        # print(f"""hdfs dfs -copyToLocal {hdfs_folderways} {local_folderways}""")
        os.system(f"""hdfs dfs -copyToLocal {hdfs_folderways} {local_folderways}""")

    def upload_folder(self, hdfs_folderways: str, local_folderways: str):
        """
        description
        ----------
        метод загрузки папки из локального хранилища до hdfs


        parameters
        ----------
        hdfs_folderways: str, путь до hdfs хранилища
        local_folderways: str, путь до локального хранилища
        """
        # print(f"""hdfs dfs -copyFromLocal {local_folderways} {hdfs_folderways}""")
        os.system(f"""hdfs dfs -mkdir -p {hdfs_folderways}""")
        os.system(f"""hdfs dfs -copyFromLocal {local_folderways} {hdfs_folderways}""")

    def load_file(self, hdfs_fileways: str, local_fileways: str):
        """
        description
        ----------
        метод загрузки файла из локального хранилища до hdfs


        parameters
        ----------
        hdfs_fileways: str, путь до hdfs хранилища
        local_fileways: str, путь до локального хранилища
        """
        pass

    def upload_file(self, hdfs_fileways: str, local_fileways: str):
        """
        description
        ----------
        метод загрузки файла из локального хранилища до hdfs


        parameters
        ----------
        hdfs_fileways: str, путь до hdfs хранилища
        local_fileways: str, путь до локального хранилища
        """
        pass
