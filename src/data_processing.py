import os
import cv2
import torch
import numpy as np  
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):
    def __init__(self, root: str, device: str = "cpu"):
        self.root = root
        self.image_path = os.path.join(root, "Images")
        self.label_path = os.path.join(root, "Labels")
        self.device = device
        
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label directory not found: {self.label_path}")
        
        valid_exts = (".jpg", ".jpeg", ".png")
        all_images = [f for f in sorted(os.listdir(self.image_path)) if f.lower().endswith(valid_exts)]
        self.img_name = [
            f for f in all_images
            if os.path.exists(os.path.join(self.label_path, str(f).rsplit('.', 1)[0] + ".txt"))
        ]
        self.label_name = [f for f in sorted(os.listdir(self.label_path)) if f.lower().endswith(".txt")]
        
        logger.info(f"Data Processing Initialized with {len(self.img_name)} paired image-label samples.")

    
    def __getitem__(self, idx):
        try:
            img_file = self.img_name[idx]
            image_path = os.path.join(self.image_path, str(img_file))
            logger.info(f"Image Path : {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image at: {image_path}")
                
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = img_rgb / 255.0
            img_res = torch.as_tensor(img_res, dtype=torch.float32).permute(2, 0, 1)
            
            label_file = str(img_file).rsplit('.', 1)[0] + ".txt"
            label_path = os.path.join(self.label_path, label_file)
            
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found : {label_path}")
            
            box = []
            with open(label_path, "r") as label_file_obj:
                lines = [line.strip() for line in label_file_obj if line.strip()]
                if lines:
                    l_count = int(lines[0])
                    for line in lines[1: 1 + l_count]:
                        coords = list(map(int, line.split()))
                        if len(coords) == 4:
                            box.append(coords)
                
            target = {
                "image_id": torch.tensor([idx], dtype=torch.int64)
            }
            
            if box:
                area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
                labels = [1] * len(box)
                
                target['boxes'] = torch.tensor(box, dtype=torch.float32)
                target['area'] = torch.tensor(area, dtype=torch.float32)
                target['labels'] = torch.tensor(labels, dtype=torch.int64)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                
            img_res = img_res.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)
            return img_res, target
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data", e)
            
    def __len__(self):
        return len(self.img_name)
    
if __name__ == "__main__":
    root_path = "artifacts/raw" if os.path.exists("artifacts/raw") else "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = GunDataset(root=root_path, device=device)
    image, target = dataset[0]
    
    print("Image Shape : ", image.shape)
    print("Target Keys : ", target.keys())
    print("Bounding Boxes : ", target['boxes'])