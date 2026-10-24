# Importing packages
import pytest
import pandas as pd
from src.components.transform_data import TransformData
from src.components.config_entity import DataIngestionConfig

# Creating a fixture to give the train dataset
@pytest.fixture(scope="module")
def create_train_dataset():
    train_data_path = DataIngestionConfig().train_data_path
    train_df = pd.read_parquet(train_data_path)
    return train_df

# Verifying that the feature engineering function is working as expected
@pytest.mark.unit
def test_generate_features(create_train_dataset):
    df = create_train_dataset
    transform = TransformData()
    trans_df = transform.generate_features(df)
    assert 'soc_med_ratio' in trans_df.columns
    assert 'game_ratio' in trans_df.columns
    assert 'socmed_game_ratio' in trans_df.columns
    assert 'work_study_ratio' in trans_df.columns
    assert 'sleep_to_screen_hrs' in trans_df.columns
    assert 'weekend_to_sleep' in trans_df.columns
    assert 'screentime_sleep_diff' in trans_df.columns
    assert 'weekend_daily_diff' in trans_df.columns
    assert 'weekend_to_daily_hrs' in trans_df.columns
    assert 'sleep_int_socmed' in trans_df.columns
    assert 'age_int_screentime' in trans_df.columns
    assert 'id' not in trans_df.columns
    assert len(trans_df.columns) == 24


# Verifying that the preprocessing object is being created
@pytest.mark.unit
def test_create_preprocessing_object(create_train_dataset):
    df = create_train_dataset
    transform = TransformData()
    preprocess_obj = transform.create_preprocessing_object(df)
    assert preprocess_obj is not None

# Verifying that the data transformation process is working as expected
@pytest.mark.unit
def test_initiate_data_transformation(create_train_dataset):
    transform = TransformData()
    train_arr_comb, test_arr_comb, preprocessor_obj = transform.initiate_data_transformation(save_object=False)
    assert train_arr_comb is not None
    assert test_arr_comb is not None
    assert preprocessor_obj is not None