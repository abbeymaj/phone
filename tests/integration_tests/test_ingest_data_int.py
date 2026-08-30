# Importing packages
import pytest
import pandas as pd
import io
import requests
from src.components.config_entity import DataIngestionConfig

# Verifying that the data source is accessible
@pytest.mark.integration
def test_data_source_accessible():
    raw_data_path = DataIngestionConfig().raw_data_path
    response = requests.get(raw_data_path, timeout=10)
    assert response.status_code == 200, f"Failed to fetch file. Status code: {response.status_code}"
    assert len(response.content) > 0, "Fetched file is empty"

@pytest.mark.integration
def test_fetch_and_parse_data():  
    raw_data_path = DataIngestionConfig().raw_data_path
    response = requests.get(raw_data_path, timeout=10)
    data = pd.read_parquet(io.BytesIO(response.content))
    assert not data.empty, "Parsed data is empty"
    