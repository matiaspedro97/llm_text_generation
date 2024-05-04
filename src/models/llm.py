import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead

class CausalLLModel:
    def __init__(self, model_cfg: dict, lora_cfg: dict, bnb_cfg: dict):
        self.model_cfg = model_cfg
        self.lora_cfg = lora_cfg
        self.bnb_cfg = bnb_cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mname_key = "pretrained_model_name_or_path"
        self.bnb_dtype = "bnb_4bit_compute_dtype"

        # build configs
        self.bnb = self.get_bnb_config()
        self.lora = self.get_lora_config()

        # model and tokenizer
        self.model = None
        self.tokenizer = None

    def get_bnb_config(self):
        # get dtype in string format
        dtype_ = self.bnb_cfg.get(self.bnb_dtype)

        # assign compute dtype (in case bnb dtype was defined)
        if dtype_ is not None:
            self.bnb_cfg[self.bnb_dtype] = eval(f"torch.{dtype_}")

        # build bnb config
        bnb_config = BitsAndBytesConfig(
            **self.bnb_cfg
        )

        return bnb_config
    
    def get_lora_config(self):
        # build peft config
        peft_config = LoraConfig(**self.lora_cfg)

        return peft_config

    def build_model(self):
        # assign quantization config
        bnb_map = {'quantization_config': self.bnb}
        model_map = {**self.model_cfg, **bnb_map}

        self.model = AutoModelForCausalLM.from_pretrained(
            **model_map
        )

        self.model = get_peft_model(self.model, peft_config=self.lora)
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_map.get(self.mname_key)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        return self.model, self.tokenizer
