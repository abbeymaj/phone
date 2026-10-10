# Importing packages
import sys
import joblib
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig
from src.exception import CustomException

# Creating a class to transform the raw data
class TransformData():
    '''
    This class is responsible for transforming the raw data and preparing it
    to be stored in a feature store. The class has three methods - the 
    constructor, a method to generate engineered features and a method to 
    transform the data for storage.
    '''
    # Creating the constructor for the class
    def __init__(self):
        '''
        This is the constructor for the class. The constructor initiates the path to the train
        and test datasets.
        '''
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    
    # Creating a feature to generate the engineered features
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        This method is responsible for engineering new features from the given
        dataframe.
        ========================================================================================
        ------------------
        Parameters:
        ------------------
        df : pd.DataFrame - This is the dataframe that will be used to create the new feature.
        
        ------------------
        Returns:
        ------------------
        df : pd.DataFrame - This is the dataframe with the new feature.
        ========================================================================================
        '''
        try:
            # Dropping the 'id' column if it exists in the dataframe
            if 'id' in list(df.columns):
                df.drop(labels=['id'], axis=1, inplace=True)
            
            # Generating the features from the dataset
            df['soc_med_ratio'] = df['social_media_hours'] / df['daily_screen_time_hours']
            df['game_ratio'] = df['gaming_hours'] / df['daily_screen_time_hours']
            df['socmed_game_ratio'] = (df['gaming_hours'] + df['social_media_hours']) / df['daily_screen_time_hours']
            df['work_study_ratio'] = df['work_study_hours'] / df['daily_screen_time_hours']
            df['sleep_to_screen_hrs'] = df['daily_screen_time_hours'] / df['sleep_hours']
            df['weekend_to_sleep'] = df['weekend_screen_time'] / df['sleep_hours']
            df['screentime_sleep_diff'] = df['daily_screen_time_hours'] - df['sleep_hours']
            df['weekend_daily_diff'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
            df['weekend_to_daily_hrs'] = df['weekend_screen_time'] / df['daily_screen_time_hours']
            df['sleep_int_socmed'] = df['sleep_hours'] * df['social_media_hours']
            df['age_int_screentime'] = df['age'] * df['daily_screen_time_hours']
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    # Creating a method to create the preprocessing object
    def create_preprocessing_object(self, df:pd.DataFrame):
        '''
        This method is responsible for creating the preprocessing object. The preprocessing
        pipeline uses scikit-learn's pipelines to build the preprocessing object.
        ========================================================================================
        ------------------
        Returns:
        ------------------
        proprocessing_obj: scikit-learn Preprocessing object - This is the preprocessing object
        used to transform the data
        ========================================================================================
        '''
        try:
            # Creating the list of numerical columns
            num_cols = list(df.select_dtypes(exclude='object').columns)
            
            # Creating the list of categorical columns
            cat_cols = [x for x in list(df.columns) if x not in num_cols+['stress_level']]
            
            # Creating the list of ordinal columns
            cat_ord_cols = [x for x in list(df.columns) if x not in num_cols+cat_cols]
            
            # Creating the pipeline for numerical features
            num_pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                ('std', StandardScaler())
            ])
            
            # Creating the pipeline for categorical features
            cat_pipeline = Pipeline([
                ('impute_cat', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                ('ohe', OneHotEncoder(sparse_output=False))
            ])
            
            # Creating the pipeline for ordinal features
            cat_ord_pipeline = Pipeline([
                ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            
            # Creating the combined pipeline
            fin_pipeline = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols),
                ('cat_ord', cat_ord_pipeline, cat_ord_cols)
            ])
            
            return fin_pipeline
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Creating a method to initiate the transformation
    def initiate_data_transformation(self, save_object:bool=True):
        '''
        This method is responsible for initiating the data transformation process. This 
        method takes the paths to the training and testing datasets as inputs and performs the 
        necessary transformations on the data before it is stored in the feature store.
        ========================================================================================
        ------------------
        Parameters:
        ------------------
        save_object: bool - A flag indicating whether to save the preprocessing object.
        The default value is True.
        
        ------------------
        Returns:
        ------------------
        train_arr : Pandas Dataframe - This is the transformed train dataset.
        test_arr : Pandas Dataframe - This is the transformed test dataset.
        preprocessor_obj : scikit-learn ColumnTransformer pipeline - This is the 
        preprocessor object to transform the data.
        ========================================================================================
        '''
        try:
            pass
        
        except Exception as e:
            raise CustomException(e, sys) from e