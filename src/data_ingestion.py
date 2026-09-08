import os
import kagglehub
import shutil
import zipfile

from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir
        
    def create_raw_dir(self) -> str:
        raw_dir = os.path.join(self.target_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        return raw_dir
    
    def extract_yolo_recursive(self, dataset_path: str, raw_dir: str):
        images_dst = os.path.join(raw_dir, 'Images')
        labels_dst = os.path.join(raw_dir, "Labels")
        
        for d in [images_dst, labels_dst]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        image_exts = (".jpg", ".jpeg", ".png")
        label_exts = (".txt",)
        
        img_count, lbl_count = 0, 0
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                src = os.path.join(root, file)
                
                if file.lower().endswith(image_exts):
                    shutil.copy(src, os.path.join(images_dst, file))
                    img_count += 1
                elif file.lower().endswith(label_exts):
                    shutil.copy(src, os.path.join(labels_dst, file))
                    lbl_count += 1
        logger.info(f"Collected {img_count} images and {lbl_count} labels")
        
        if img_count == 0 or lbl_count == 0:
            raise FileNotFoundError(
                "No Images or Labels found while walking dataset structure"
            )
            
    def extract_images_and_labels(self, dataset_path: str, raw_dir: str):
        try:
            logger.info(f"Dataset Path : {dataset_path}")
            
            if dataset_path.endswith(".zip"):
                extract_dir = dataset_path.replace(".zip", "")
                if not os.path.exists(extract_dir):
                    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                dataset_path = extract_dir
            self.extract_yolo_recursive(dataset_path, raw_dir)
        except Exception as e:
            raise CustomException("Error while extracting images and labels", e)
    
    def download_dataset(self, raw_dir: str):
        try:
            if os.path.exists("data/Images") and os.path.exists("data/Labels") and len(os.listdir("data/Images")) > 0:
                logger.info("Found local data/ directory. Ingesting from local dataset...")
                self.extract_images_and_labels("data", raw_dir)
            else:
                dataset_path = kagglehub.dataset_download(self.dataset_name)
                logger.info(f"Dataset Download at : {dataset_path}")
                self.extract_images_and_labels(dataset_path, raw_dir)
        except Exception as e:
            raise CustomException("Error while downloading dataset", e)
            
    def run(self):
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data Ingestion completed Successfully")
        except Exception as e:
            raise CustomException("Data Ingestion Pipeline failed", e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    data_ingestion.run()