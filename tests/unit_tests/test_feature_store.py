# Importing packages
import os
import pytest
import pandas as pd
import pyfakefs as fs
from unittest.mock import MagicMock
from src.components.create_feature_store import CreateFeatureStore

# Creating a fixture to mock to the pandas to_parquet function
@pytest.fixture
def mock_parquet_saver(mocker, fs):
    def _side_effect_save(path, **kwargs):
        fs.create_file(path, contents="dummy parquet data")
    return mocker.patch("pandas.DataFrame.to_parquet", side_effect=_side_effect_save)

# Verifying that the feature store folder can be successfully created
@pytest.mark.unit
def test_create_feature_store(mocker, fs, mock_parquet_saver):
    # Creating dummy dataframes
    df_train = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df_test = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    
    # Mocking the configuration object
    mock_config = MagicMock()
    mock_config.xform_train_path = "/fake/path/train.parquet"
    mock_config.xform_test_path = "/fake/path/test.parquet"
    
    # Creating the expected call objects
    expected_train = mocker.call(mock_config.xform_train_path, index=False, compression="zstd")
    expected_test = mocker.call(mock_config.xform_test_path, index=False, compression="zstd")
    
    # Manually injecting the mock configuration
    pipeline = CreateFeatureStore()
    pipeline.feature_store_config = mock_config
    
    # Executing the method
    train_path, test_path = pipeline.create_feature_store(df_train, df_test)
    
    # Running the assertions
    assert os.path.exists("/fake/path")
    assert mock_parquet_saver.call_count == 2
    assert os.path.isfile(train_path)   
    assert os.path.isfile(test_path)
    
