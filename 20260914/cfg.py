"""
    参数模块
    
"""

from pathlib import Path
path = Path(__file__).parent

print(path)

class Config:
    FILE_PATH = path / "data" / "train.txt"
    DEV_FILE_PATH = path / "data" / "dev.txt"

print(Config.FILE_PATH)







