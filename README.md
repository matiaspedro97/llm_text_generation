llm_text_generation
==============================

Exploring the text generation capabilities of the most recent Large Language Models.

**LLM_TEXT_GENERATION** is a comprehensive project aimed at developing and fine-tuning large language models (LLMs) for text generation tasks. This repository includes configurations, data handling, model training, pipeline management, and visualization tools to facilitate the development and deployment of LLMs.

Project Organization
------------

    LLM_TEXT_GENERATION/
    ├── configs/
    │   └── pipeline/
    ├── data/
    ├── docs/
    ├── little-llama-ft-qa/
    ├── little-llama2-ft-qa/
    ├── little-llama2-ft-summarize/
    ├── models/
    ├── notebooks/
    ├── references/
    ├── reports/
    ├── requirements/
    ├── src/
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── .gitkeep
    │   │   └── load.py
    │   ├── features/
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── .gitkeep
    │   │   ├── llm.py
    │   │   └── train.py
    │   ├── pipeline/
    │   │   ├── __init__.py
    │   │   ├── pipe_train.py
    │   │   ├── runs/
    │   │   │   ├── run_inference_example.py
    │   │   │   └── run_train_ft.py
    │   ├── visualization/
    │   │   ├── __init__.py
    │   │   ├── .gitkeep
    │   │   ├── visualize.py
    │   ├── config.py
    │   ├── constants.py
    ├── src.egg-info/
    ├── .env
    ├── .gitignore
    ├── environment.yml
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── setup.py
    ├── test_environment.py
    └── tox.ini

--------


## Setup Environment
To setup the environment, you should build two distinct virtual environment: (1) conda environment (with basic ML project packages) and (2) virtualenv (with essencial packages to run the ML pipelines)

#### Note: Make sure to install Anaconda or Miniconda

## 1. Clone the repository
```bash
git clone https://github.com/yourusername/LLM_TEXT_GENERATION.git
```

#### 2. Setup the Conda enviroment
```bash
conda create env -file environment.yml
```

#### 3. Activate the new environment
```bash
conda deactivate

conda activate python3.8
```

#### 4. Build the dev virtualenv on top of the conda environment
```bash
virtualenv .venv-dev
```

#### 5. Activate the dev virtualenv

```bash
. .venv-dev/bin/activate  # linux
```
OR 
```bash
.venv-dev/Scripts/activate  # windows
```

#### 6. Install the dependencies
With both environments activated, install hard dependencies
```bash
pip install -r requirements/requirements.txt  # windows/linux
```

#### 7. You are now able to run the scripts

```bash
# fine-tuning an existing LLM from a config file
python src/models/train.py

# Inference example from config file
python src/pipeline/runs/run_inference_example.py
```

## Project Source Code details
- [Configs](configs/): Contains configuration files.
  - [Pipeline-Configs](configs/pipeline/): Pipeline-specific configurations.
- [Data](data/): Placeholder for data storage.
- [Documentation](docs/): Documentation files.
- [Models](models/): Placeholder for model-related files.
- [Notebooks](notebooks/): Jupyter notebooks for experiments and prototyping.
- [References](references/): Reference materials and papers.
- [Reports](reports/): Dissemination and experiment reports.
- [Requirements](requirements/): Dependency requirements.
- [Data](src/data/): Data parsing and loading scripts.
- [Features](src/features/): Feature engineering scripts.
- [Models](src/models/): Model definition and training scripts.
- [Pipeline](src/pipeline/): Pipeline scripts.
- [Runs](src/runs/): Example run scripts.
- [Visualization](src/visualization/): Visualization scripts.

## Reports
Please check the Reports folder to see some of the obtained results: [Reports](reports/)

## Configuration
Please check the pipeline configuration files. You'll need to define one to run the experiment scripts: [Configs](configs/)

## Contributions
Contributions are welcomed! If you want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request detailing your changes.

## License

This project is licensed under the [MIT License](LICENSE).  

## Contact
For any questions or inquiries, please contact matiaspedro97@gmail.com

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
