from transformers import PretrainedConfig

class MCTConfig(PretrainedConfig):
    model_type = "mct_transformer"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        p_drop=0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.p_drop = p_drop # Linked to the stochastic stem dropout

    @classmethod
    def small(cls):
        """Small configuration (125M parameters)"""
        return cls(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)

    @classmethod
    def medium(cls):
        """Medium configuration (350M parameters)"""
        return cls(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)

    @classmethod
    def large(cls):
        """Large configuration (1B parameters)"""
        return cls(hidden_size=1536, num_hidden_layers=24, num_attention_heads=24)
