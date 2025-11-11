import logging 
import os  # For creating directories to store logs
from datetime import datetime 

LOGS_DIR = "logs" 
os.makedirs(LOGS_DIR, exist_ok= True)

# Organizing log by day
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")


logging.basicConfig(
    filename= LOG_FILE,
    format= '%(asctime)s - %(levelname)s - %(message)s',   #levelname = {INFO, WARNING, ERROR, customized ones}
    level = logging.INFO # dictate that only {INFO, WARNING, ERROR} will show, other things lower than INFO will not show
)

def get_logger(name):
    '''
    Initializing the logger
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger