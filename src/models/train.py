from torch.utils.data import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

class CausalLMTrainer:
    def __init__(self, **train_args):
        # config params
        self.train_args = train_args

        # build train config
        self.train_cfg = self.define_training_arguments()

        # trainer
        self.trainer_obj = None

    def define_training_arguments(self):
        # build train config
        training_arguments = TrainingArguments(
            **self.train_args
        )

        return training_arguments

    def build_trainer(self, model, tokenizer, train_dset, val_dset, peft_cfg):
        # get trainer
        self.trainer_obj = SFTTrainer(
            model=model,
            train_dataset=train_dset,
            eval_dataset=val_dset,
            peft_config=peft_cfg,
            dataset_text_field="text",
            max_seq_length=4096,
            tokenizer=tokenizer,
            args=self.train_cfg
        )

    def fit(self):
        # trainer
        self.trainer_obj.train()

    def push_to_hf(self):
        self.trainer_obj.push_to_hub()
