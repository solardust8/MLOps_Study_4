# MLOps Study / Part 3 (HDFS + Hashicorp Vault)

MLOps study repository to practice ML project organization and deployment. 
This will be dedicated to a simple spam detection service using sequence classification with BERT.
Made by Domnitsky Egor (M4130)


# Task overview

I was assigned with a task to develop simple spam/ham message type predictor service using  any model for sequence classification trained on related data. Main objective was to practice with DevOps tools, including Git repostitory preparation, familiarizing with DVC to version control the data, developing a simple (the most basic) API to reach the model, and building the Docker container to wrap all things up. Also, it was important to provide this repository with simple CI/CD pipeline to automaticly control the assembly of distributive and building of the container in CI as well as running simple tests inside of the container in CD. 

## Data 

The dataset attached to the task was [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). As seen in `notebooks/data_investigation_and_prep.ipynb` dataset was severly imbalanced. So, in order to upsample the minor class `nlpaug` was used (see `notebooks/data_augmentation_and_model_finetuning.ipynb` for info on model finetuning with data augmentation, as well as for required dataset structure). 

To get the augmented dataset, use `dvc pull` inside of the repo root (**note**: dvc repo is stored in public Google Drive Folder, you will have to authenticate to pull the files).  


## Model

To classify spam/ham messages [DistilBert for Sequence Classification by Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert) was used. 

Author is aware that using a deep encoder model for such trivial task might be an overkill. Nevertheless, this was made intentionally for practicing with deep language models and later maintaining such in containers (installing and configuring **torch + transformers** in a container). 

## Requirements

Run `pip install -r requirements.txt`

## Dockerfile

**IMPORTANT**: .safetensors files are not stored neither in Git LFS nor in DVC repo, but in [Google Drife weights folder](https://drive.google.com/drive/folders/1rabrFtLligrzg1MyzYHPqreOURJ5Pq1S?usp=sharing). Before building, consider manually downloading the safetensors files and placing them to `weigths/<model_name>` folders together with `config.json` files.

## Primitive API

API utilizes FastAPI lib. Server is ran with `uvicorn src.run:app --host 0.0.0.0 --port 8000`. See more at [FastAPI page](https://fastapi.tiangolo.com/).

To get the result, you need to send a GET /predict request with json payload, containing `"text"` field with a list of prompts no longer that 256 tokens. See `sample_query.json` for an example.
