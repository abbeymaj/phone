# Importing packages
import os
from dataclasses import dataclass

# Creating a dataclass to store the paths to the raw dataset, 
# training and test datasets.
@dataclass
class DataIngestionConfig():
    raw_data_path: str = "https://raw.githubusercontent.com/abbeymaj80/my-ml-datasets/master/project_datasets/phone_addiction/train1.parquet"
    train_data_path: str = os.path.join('artifacts', 'train.parquet')
    test_data_path: str = os.path.join('artifacts', 'test.parquet')