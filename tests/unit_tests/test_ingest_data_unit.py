# Importing packages
import pytest
import pandas as pd
from src.components.config_entity import DataIngestionConfig
from src.components.ingest_data import IngestData

# Creating a fixture to read the raw data
@pytest.fixture
def read_raw_data():
    df = pd.read_parquet(DataIngestionConfig().raw_data_path)
    return df

# Verifying that the raw data can be read correctly
@pytest.mark.unit
def test_read_raw_data(read_raw_data):
    df = read_raw_data
    assert df is not None
    
# Verifying that dataframe is a pandas dataframe
@pytest.mark.unit
def test_is_dataframe(read_raw_data):
    df = read_raw_data
    assert isinstance(df, pd.DataFrame)

# Verifying that the initiate data ingestion function works as expected
@pytest.mark.unit
def test_initiate_data_ingestion():
    ingestion = IngestData()
    _, _, train_df, test_df = ingestion.initiate_data_ingestion(create_artifacts_folder=False, save_objects=False)
    assert train_df is not None
    assert test_df is not None

# Verifying that the train and test sets are pandas dataframes
@pytest.mark.unit
def test_train_and_test_is_dataframe():
    ingestion = IngestData()
    _, _, train_df, test_df = ingestion.initiate_data_ingestion(create_artifacts_folder=False, save_objects=False)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)