# Importing packages
from src.components.ingest_data import IngestData

# Running the pipeline
if __name__ == "__main__":
    ingest_data = IngestData()
    train_data_path, test_data_path = ingest_data.initiate_data_ingestion()