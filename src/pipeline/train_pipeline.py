from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformer
from src.component.model_trainer import ModelTrainer

if __name__ == "__main__":
    
    obj=DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    data_trans = DataTransformer()
    train_arr,test_arr,_=data_trans.initiate_data_transformer(train_path,test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))


