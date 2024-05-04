import os
import dotenv

# project directory
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)

# load env variables
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

# datasets
data_dir = os.path.join(project_dir, 'data')
data_raw_dir = os.path.join(data_dir, 'raw')
data_proc_dir = os.path.join(data_dir, 'processed')

# configs
config_dir = os.path.join(project_dir, 'configs')
pipeline_dir = os.path.join(config_dir, 'pipeline')