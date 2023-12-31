import MySQLdb
from app import settings


class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance = MySQLdb.connect(
                **settings.DATABASES
            )
        return cls._instance

    @classmethod
    def startTransaction(cls):
        if not cls._instance:
            cls()
        cls._instance.cursor().execute("START TRANSACTION")

    @classmethod
    def commitTransaction(cls):
        if not cls._instance:
            cls()
        cls._instance.commit()
        cls._instance.cursor().close()

    # def rollbackTransaction(self):
    #     if not self._instance:
    #         self._instance.rollback()
