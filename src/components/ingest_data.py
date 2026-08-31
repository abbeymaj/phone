# Importing packages
import sys
import os
import pandas as pd
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.model_selection import train_test_split
from src.components.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


# Creating a class to ingest the data
class IngestData():
    '''
    This class is used to ingest the raw data from the source. The data is split
    into a train and test set for further processing. The train and test data
    is then stored in the artifacts directory.
    '''
    # Creating the constructor for the class
    def __init__(self):
        '''
        This is the constructor of the IngestData class. It instantiates the data ingestion
        config for further use.
        '''
        self.data_ingestion_config = DataIngestionConfig()
    
    # Creating a method to ingest the data
    def initiate_data_ingestion(
        self, 
        create_artifacts_folder: bool = True, 
        save_objects: bool =True,
        return_datasets: bool = False
        ):
        '''
        This method will ingest the data from source, split the dataset into a train
        and test dataset. The function will also create the artifacts folder and store
        the train and test dataset in the artifacts folder.
        ====================================================================================
        ---------------
        Returns:
        ---------------
        train file path : str - This is the path to the train dataset.
        test file path : str - This is the path to the test dataset.
        ====================================================================================
        '''
        try:
            # Creating the artifacts folder if it does not already exist
            if create_artifacts_folder:
                os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            
            # Reading the raw data from the source into a dataframe
            df = pd.read_parquet(self.data_ingestion_config.raw_data_path)
            
            # Splitting the data into the train and test sets
            train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['addicted_label'])
            
            # Storing the train and test datasets into the artifacts folder
            if save_objects:
                train_df.to_parquet(self.data_ingestion_config.train_data_path, index=False, compression='zstd')
                test_df.to_parquet(self.data_ingestion_config.test_data_path, index=False, compression='zstd')
            
            if return_datasets:
                return (
                    self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path,
                    train_df,
                    test_df
                )
            else:
                return (
                    self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path
                )
            
        except Exception as e:
            raise CustomException(e, sys) from e
    