"""
Main Training Script for Vision-Language Model
Supports single-GPU and multi-GPU (DDP) training
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models import VisionLanguageModel
from data.dataset import COCOCaptionDataset, collate_fn


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


class Trainer:
    """Training orchestrator"""
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Build data
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision
        self.use_amp = config['hardware']['mixed_precision'] in ['fp16', 'bf16']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def _build_model(self):
        model_config = self.config['model']
        
        model = VisionLanguageModel(
            vision_model_name=model_config['vision_encoder']['name'],
            language_model_name=model_config['language_model']['name'],
            projection_hidden_dim=model_config['projection']['hidden_dim'],
            projection_num_layers=model_config['projection']['num_layers'],
            freeze_vision=model_config['vision_encoder']['freeze'],
            freeze_language_layers=model_config['language_model']['freeze_layers'],
            dropout=model_config['projection']['dropout']
        )
        
        return model
    
    def _build_dataloaders(self):
        data_config = self.config['data']
        train_config = self.config['training']
        dataloader_config = self.config['dataloader']
        phase_config = self.config['phase1']
        
        # Build datasets
        train_dataset = COCOCaptionDataset(
            root_dir=data_config['coco_dir'],
            split=phase_config['train_split'],
            vision_processor=self.raw_model.vision_encoder.processor,
            tokenizer=self.raw_model.tokenizer,
            max_caption_length=phase_config['max_caption_length'],
            return_all_captions=False
        )
        
        val_dataset = COCOCaptionDataset(
            root_dir=data_config['coco_dir'],
            split=phase_config['val_split'],
            vision_processor=self.raw_model.vision_encoder.processor,
            tokenizer=self.raw_model.tokenizer,
            max_caption_length=phase_config['max_caption_length'],
            return_all_captions=True
        )
        
        # Samplers
        train_sampler = None
        val_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            collate_fn=collate_fn,
            persistent_workers=dataloader_config['persistent_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['validation']['val_batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            collate_fn=collate_fn
        )
        
        if self.is_main:
            print(f"✓ Train dataset: {len(train_dataset)} samples")
            print(f"✓ Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _build_optimizer(self):
        train_config = self.config['training']
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon']
        )
        
        return optimizer
    
    def _build_scheduler(self):
        train_config = self.config['training']
        
        num_training_steps = len(self.train_loader) * train_config['num_epochs']
        num_warmup_steps = train_config['warmup_steps']
        
        from transformers import get_cosine_schedule_with_warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_epoch(self, epoch):
        self.model.train()
        
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                if self.config['training']['max_grad_norm'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config['training']['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.is_main:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / num_batches,
                    'lr': self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation", disable=not self.is_main)
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
            
            if self.is_main:
                pbar.set_postfix({'val_loss': loss.item()})
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filename):
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        if self.is_main:
            print(f"✓ Saved checkpoint: {filename}")
    
    def train(self):
        num_epochs = self.config['training']['num_epochs']
        
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"STARTING TRAINING")
            print(f"{'='*60}")
            print(f"Epochs: {num_epochs}")
            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
            print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            if self.is_main:
                print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"✓ New best model! Val loss: {val_loss:.4f}")
                
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    base_config_path = Path(args.config).parent / 'base_config.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    config = {**base_config, **config}
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Train
    trainer = Trainer(config, rank=rank, world_size=world_size)
    trainer.train()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()