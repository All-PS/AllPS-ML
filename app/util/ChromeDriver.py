from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class ChromeDriver:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")

            service = Service(executable_path=ChromeDriverManager().install())
            cls._instance = webdriver.Chrome(service=service, options=options)
        return cls._instance
