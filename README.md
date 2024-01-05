# cookie_test

A short description of the project.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project using the user defined <project_name>
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


Building the docker image from the trainer.dockerfile:
```
docker build -f trainer.dockerfile . -t trainer:latest
```
Running an experiment in the container:
```
docker run --name experiment1 trainer:latest
```
Starting an interactive session in the container with shell (can use bash as well). Note, this opens a new container, and isn't a equivalent to mounting, ie. files are not transferred. Only for structural idea of container:
```
docker run -it --entrypoint sh {image_name}:{tag}
```
Copying files (if not just a mount):
```
docker cp {container_name}:{dir_path}/{file_name} {local_dir_path}/{local_file_name}
```
Mounting a volume is preferred over the above, and is done by the below. Depending on the OS, this may change from %cd& to $pwd or {$PWD}, see [here](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10):
```
docker run --name {container_name} -v %cd%/models:/models/ trainer:latest
```
And example of mounting multiple files was done using the predict.dockerfile, which image was built and the following container ran:
```
docker run --name predict --rm -v %cd%/models/trained_model_50_1e-03.pt:/models/trained_model_50_1e-03.pt -v %cd%/data/testloader.pt:/testloader.pt predict:latest evaluate trained_model_50_1e-03.pt
```
So we run the container remotely, mount the appropriate files, and use the built docker container we've established, then the python code to execute on the remote terminal.

