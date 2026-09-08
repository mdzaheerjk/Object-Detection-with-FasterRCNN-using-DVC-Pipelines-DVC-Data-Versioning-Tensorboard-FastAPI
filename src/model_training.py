import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "models")

class ModelTraining:
    def __init__(
        self,
        model_class=FasterRCNNModel,
        num_classes: int = 2,
        learning_rate: float = 0.005,
        epochs: int = 10,
        dataset_path: str = "artifacts/raw",
        device: str = "cpu",
        model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
        batch_size: int = 1,
        learing_rate: float = None
    ):
        if learing_rate is not None:
            learning_rate = learing_rate
            
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device
        self.model_save_path = model_save_path
        self.batch_size = batch_size

        os.makedirs(self.model_save_path, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(PROJECT_ROOT, "tensorboard_logs", timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
            logger.info("TensorBoard SummaryWriter not available; proceeding with standard logging.")
        
        try:
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(self.device)
            logger.info("Model moved to device")
            
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.33)
            logger.info("SGD Optimizer and StepLR scheduler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize model training {e}")
            raise CustomException("Failed to initialize model training", e)
        
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            actual_dataset_path = self.dataset_path
            if not os.path.exists(actual_dataset_path):
                if os.path.exists(os.path.join(PROJECT_ROOT, self.dataset_path)):
                    actual_dataset_path = os.path.join(PROJECT_ROOT, self.dataset_path)
                elif os.path.exists(os.path.join(PROJECT_ROOT, "data")):
                    actual_dataset_path = os.path.join(PROJECT_ROOT, "data")
                elif os.path.exists("data"):
                    actual_dataset_path = "data"
                    
            dataset = GunDataset(actual_dataset_path, self.device)
            train_size = int(0.85 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
            
            logger.info(f"Dataset split: {train_size} train samples, {val_size} validation samples.")
            return train_loader, val_loader
        
        except Exception as e:
            logger.error(f"Failed to split data {e}")
            raise CustomException("Failed to split data", e)
        
    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()
            best_val_loss = float('inf')
            
            for epoch in range(1, self.epochs + 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Starting epoch {epoch}/{self.epochs} (LR: {current_lr:.6f})")
                self.model.train()
                epoch_train_loss = 0.0
                
                for i, (images, targets) in enumerate(train_loader):
                    images = [img.to(self.device) for img in images]
                    targets = [{key: val.to(self.device) for key, val in target.items()} for target in targets]
                    
                    self.optimizer.zero_grad()
                    losses = self.model(images, targets)
                    
                    if isinstance(losses, dict):
                        total_loss = sum(losses.values())
                    elif isinstance(losses, (list, tuple)):
                        total_loss = sum(losses)
                    else:
                        total_loss = losses
                        
                    total_loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    
                    loss_val = total_loss.item()
                    epoch_train_loss += loss_val
                    global_step = (epoch - 1) * len(train_loader) + i
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/Train_Batch", loss_val, global_step)
                        
                    del losses, total_loss, images, targets
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                self.lr_scheduler.step()
                avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
                logger.info(f"Epoch {epoch} - Average Train Loss: {avg_train_loss:.4f}")
                if self.writer is not None:
                    self.writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
                    self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                    self.writer.flush()
                
                # Validation loss computation
                val_loss_total = 0.0
                with torch.no_grad():
                    for val_images, val_targets in val_loader:
                        val_images = [img.to(self.device) for img in val_images]
                        val_targets = [{key: val.to(self.device) for key, val in target.items()} for target in val_targets]
                        
                        self.model.train()
                        val_loss_dict = self.model(val_images, val_targets)
                        if isinstance(val_loss_dict, dict):
                            val_batch_loss = sum(val_loss_dict.values())
                        elif isinstance(val_loss_dict, (list, tuple)):
                            val_batch_loss = sum(val_loss_dict)
                        else:
                            val_batch_loss = val_loss_dict
                        val_loss_total += val_batch_loss.item()
                        
                        del val_loss_dict, val_batch_loss, val_images, val_targets
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                avg_val_loss = val_loss_total / max(len(val_loader), 1)
                logger.info(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss:.4f}")
                if self.writer is not None:
                    self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
                    self.writer.flush()
                
                model_path = os.path.join(self.model_save_path, "fasterrcnn.pth")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved successfully at: {model_path}")
                
            if self.writer is not None:
                self.writer.close()
                
        except Exception as e:
            logger.error(f"Failed to train model {e}")
            raise CustomException("Failed to train model", e)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = "artifacts/raw" if os.path.exists("artifacts/raw") else "data"
    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,
        learning_rate=0.005,
        dataset_path=dataset_path,
        device=device,
        batch_size=2,
        epochs=10
    )
    training.train()