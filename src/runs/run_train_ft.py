import os

from src.pipeline.pipe_train import PipelineTG
from src.constants import CONFIG_PATH

# pipeline QA config
config_pth = os.path.join(CONFIG_PATH, 'pipeline', 'train_ft_tinyllama2.json')

# instantiate QA pipeline loader
pipe_tg = PipelineTG(config_path=config_pth)

# run pipeline
pipe_tg.run()