# Importing packages
import os
import sys
import pandas as pd
from src.components.config_entity import FeatureStoreConfig
from src.exception import CustomException

# Creating a class to build the feature store and save the 
# transformed datasets.
class CreateFeatureStore():
    """
    This class is responsible for creating the feature store and then saving the 
    transformed datasets into the feature store. The class has two methods - 
    a constructor and a method to save the transformed datasets into the feature store.
    """
    # Creating the constructor for this class
    def __init__(self):
        """
        This is the constructor for the feature store creation
        class.
        """
        self.feature_store_config = FeatureStoreConfig()
    
    # Creating a method to save the transformed datasets into the feature store
    def create_feature_store(self, train_data:pd.DataFrame, test_data:pd.DataFrame):
        """
        This method creates the feature store and then stores the transformed datasets in the 
        feature store folder.
        =========================================================================================
        ----------------
        Parameters:
        ----------------
        train_data : pandas dataframe - This is the train dataset.
        test_data : pandas dataframe - This is the test dataset.
        
        ----------------
        Returns:
        ----------------
        transformed train data path : str - Returns the path to the transformed train dataset.
        transformed test data path : str - Returns the path to the transformed test dataset.
        =========================================================================================
        """
        try:
            # Creating the feature store directory if the directory does not exist
            dir_path = os.path.dirname(self.feature_store_config.xform_train_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Storing the transformed train and test datasets into the feature store
            train_data.to_parquet(self.feature_store_config.xform_train_path, index=False, compression='zstd')
            test_data.to_parquet(self.feature_store_config.xform_test_path, index=False, compression='zstd')
            
            return (
                self.feature_store_config.xform_train_path,
                self.feature_store_config.xform_test_path
            )
        
        except Exception as e:
            raise CustomException(e, sys) from e