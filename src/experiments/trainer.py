import torch
import logging
from transformers import Trainer, TrainingArguments, AdamW, get_cosine_schedule_with_warmup
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MCTTrainer(Trainer):
    """Custom trainer for MCT models with MCT-specific logging."""
    
    def __init__(self, *args, tokenizer=None, **kwargs):
        """
        Args:
            tokenizer: MCT tokenizer for logging morphology rate
            *args, **kwargs: Arguments passed to parent Trainer
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.mct_metrics_log = {}
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and scheduler with MCT hyperparameters.
        Hyperparameters from paper: Beta1=0.9, Beta2=0.98 [cite: 169-171]
        """
        # Create optimizer with paper's hyperparameters
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),  # From paper
            eps=1e-8,
            weight_decay=self.args.weight_decay
        )
        
        # Linear warmup (4k steps) followed by cosine decay
        # [cite: 167-168]
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(4000, num_training_steps // 10),
            num_training_steps=num_training_steps
        )
        
        logger.info("Optimizer and scheduler configured:")
        logger.info(f"  - Optimizer: AdamW with beta1=0.9, beta2=0.98")
        logger.info(f"  - Scheduler: Cosine decay with warmup")
        
        return self.optimizer, self.lr_scheduler

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step.
        Custom logic to track MCT-specific metrics (e.g., stem-dropout rate).
        """
        loss = super().training_step(model, inputs)
        
        # Log MCT-specific metrics if tokenizer available
        if self.tokenizer and hasattr(self.tokenizer, 'get_stats'):
            stats = self.tokenizer.get_stats()
            
            # Log to wandb if available
            if self.state.global_step % self.args.logging_steps == 0:
                for key, value in stats.items():
                    if isinstance(value, float):
                        self.log({f"mct_{key}": value})
                
                logger.debug(f"MCT Stats at step {self.state.global_step}: {stats}")
        
        return loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics including MCT-specific ones.
        """
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"Step {self.state.global_step}: {logs}")
        
        super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Run evaluation with MCT metric logging.
        """
        logger.info(f"Starting evaluation at step {self.state.global_step}")
        
        results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Log MCT-specific evaluation metrics
        if self.tokenizer and hasattr(self.tokenizer, 'get_stats'):
            stats = self.tokenizer.get_stats()
            for key, value in stats.items():
                if isinstance(value, float):
                    results[f"mct_{key}"] = value
        
        return results


def create_mct_trainer(model, train_dataset, eval_dataset=None, tokenizer=None, 
                      output_dir: str = "./results", 
                      num_train_epochs: int = 3,
                      per_device_train_batch_size: int = 64,
                      per_device_eval_batch_size: int = 64,
                      learning_rate: float = 6e-4,
                      weight_decay: float = 0.01,
                      **kwargs) -> MCTTrainer:
    """
    Create an MCTTrainer with standard MCT hyperparameters from the paper.
    
    Args:
        model: MCT model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: MCT tokenizer for logging
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size for training
        per_device_eval_batch_size: Batch size for evaluation
        learning_rate: Learning rate (default: 6e-4 for small model) [cite: 158]
        weight_decay: Weight decay for regularization [cite: 160]
        **kwargs: Additional arguments
        
    Returns:
        Configured MCTTrainer instance
    """
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 8),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=4000,  # From paper
        logging_steps=100,
        eval_steps=5000 if eval_dataset else None,
        save_steps=5000,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        optim="adamw_torch",
        seed=42,
        **kwargs
    )
    
    trainer = MCTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    return trainer
