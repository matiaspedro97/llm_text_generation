import json
import pydoc

from typing import List
from loguru import logger

from src.pipeline import PipelineGen

class PipelineTG(PipelineGen):
    def __init__(self, config_path: str = None, config_dict: dict = None) -> None:
        # Config path
        self.config_path = config_path

        # load modules
        if isinstance(config_dict, dict):
            config_args = self.load_modules_from_dict(config_dict)
        else:
            config_args = self.load_modules_from_json(config_path)

        # load gen attributes
        super().__init__(**config_args)

    def load_modules_from_json(self, json_path: str):
        config = json.load(open(json_path, 'r'))
        return self.load_modules_from_dict(config)

    def load_modules_from_dict(self, config: dict):    
        # Kwargs
        config_gen_args = {
            k: v 
            for k, v in config.items() 
            if k != 'modules'
        }

        # Loading class modules
        for module_name, module in config['modules'].items():
            class_ = pydoc.locate(f"src.{module['class_']}")
            params_ = module['params_']

            try:
                obj = class_(**params_)
                logger.info(f"Module {module_name} successfully loaded")
            except Exception as e:
                logger.info(f"Module {module_name} not loaded correctly." 
                             f"Please check the error:\n{e}")
                obj = None

            # assign to class attribute
            config_gen_args[module_name] = obj
            #exec(f"self.{module_name} = obj")
            

        return config_gen_args

    def run(self, push_to_hub: bool = False):
        # questions
        dataset = self.loader.process()
        train, val = dataset['train'], dataset['validation']

        # most similar documents
        model, tokenizer = self.model.build_model()
        peft_cfg = self.model.lora

        # prepare for training
        self.trainer.build_trainer(model, tokenizer, train, val, peft_cfg)

        # train
        self.trainer.fit()

        # push to hub
        if push_to_hub:
            self.trainer.push_to_hf()

        return model, tokenizer

