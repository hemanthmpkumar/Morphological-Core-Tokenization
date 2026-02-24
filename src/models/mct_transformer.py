import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_mct import MCTConfig
import logging

logger = logging.getLogger(__name__)


class MCTTransformer(PreTrainedModel):
    """
    MCT Transformer model with morphological-aware input processing.
    Supports atomic stem preservation and affix segmentation.
    """
    
    config_class = MCTConfig

    def __init__(self, config):
        super().__init__(config)
        
        logger.info("Initializing MCT Transformer")
        logger.info(f"  - Vocab size: {config.vocab_size}")
        logger.info(f"  - Hidden size: {config.hidden_size}")
        logger.info(f"  - Num layers: {config.num_hidden_layers}")
        logger.info(f"  - P_drop (morphology dropout): {config.p_drop}")
        
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Token type embeddings (distinguish stems from affixes)
        self.token_type_embeddings = nn.Embedding(3, config.hidden_size)  # 0: stem, 1: prefix, 2: suffix
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        
        # Standard Transformer Encoder Layers
        # Note: MCT doesn't change the architecture, just the input granularity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            activation="gelu",
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers
        )
        
        # Output head for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights manually (simpler than parent's complex init_weights)
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, 
                position_ids=None, return_dict=True):
        """
        Forward pass for MCT model.
        
        Args:
            input_ids: Token IDs (sequence_length,)
            attention_mask: Attention mask (sequence_length,)
            token_type_ids: Token type IDs (0=stem, 1=prefix, 2=suffix)
            labels: Labels for language modeling loss
            position_ids: Position IDs
            return_dict: Return dict or tuple
            
        Returns:
            ModelOutput with loss and logits
        """
        # Default values
        batch_size, seq_length = input_ids.shape if len(input_ids.shape) > 1 else (1, len(input_ids))
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            if len(position_ids.shape) == 1:
                position_ids = position_ids.unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids)  # (batch, seq_len, hidden)
        
        # Add position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embedding_output = embedding_output + position_embeddings
        
        # Add token type embeddings (stem vs affix distinction)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_output = embedding_output + token_type_embeddings
        
        embedding_output = self.dropout(embedding_output)
        
        # Convert attention mask to causal format for transformer
        if len(attention_mask.shape) == 2:
            # Convert (batch, seq_len) to (batch, 1, seq_len, seq_len)
            attention_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
        
        # Transformer encoder
        encoder_output = self.encoder(
            embedding_output,
            src_key_padding_mask=attention_mask[:, 0, 0, :] if len(attention_mask.shape) == 4 else (attention_mask == 0)
        )
        
        # Language model head
        logits = self.lm_head(encoder_output)  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            return (loss, logits) if loss is not None else logits
        
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )
    
    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.embeddings
    
    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.embeddings = value
    
    def get_output_embeddings(self):
        """Get output embeddings (lm_head)."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (lm_head)."""
        self.lm_head = new_embeddings

# Backward compatibility alias
MCTModel = MCTTransformer