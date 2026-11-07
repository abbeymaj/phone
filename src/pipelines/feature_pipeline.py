# Importing packages
from src.components.ingest_data import IngestData
from src.components.transform_data import TransformData
from src.components.create_feature_store import CreateFeatureStore

# Running the pipeline
if __name__ == "__main__":
    # Ingesting the data from the source files
    ingest_data = IngestData()
    train_data_path, test_data_path = ingest_data.initiate_data_ingestion()
    
    # Transforming the ingested data
    transform_data = TransformData()
    train_data, test_data, _ = transform_data.initiate_data_transformation()
    
    # Creating the feature store and storing the transformed
    # data in the feature store
    feature_store = CreateFeatureStore()
    feature_store.create_feature_store(train_data, test_data)
    