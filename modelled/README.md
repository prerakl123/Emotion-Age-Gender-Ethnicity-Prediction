# Emotion-Age-Gender-Ethnicity Prediction
This repository contains code and resources for a project that predicts a person's emotion, age, gender, and ethnicity from a given image.

> **Note**:  This type of technology can be biased and may not always be accurate. It's important to use such tools 
> responsibly and be aware of their limitations.


# Setting Up The Project
You can either set this project up in your own local environment or you can use the pre-trained models for prediction.

## Training in Your Local Environment

### Download the Dataset:
- [Facial-Emotion-Recognition-2013 (FER-2013)](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)
- [Age-Gender-Ethnicity Dataset](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv)
> **NOTE:** 
> 1. For the FER dataset, only the `fer2013.csv` file is required.
> 2. Place both the CSV files in the `dataset` directory.

### Setting up Your Virtual Environment

Create a new directory for your virtual environment folder.
```shell
$ mkdir ".venv"
$ cd ".venv"
```

Make sure the `virtualenv` library is installed.
```shell
$ pip install -U virtualenv
```

Create and activate the virtual environment.
```shell
$ venv .
$ ./Scripts/activate
```

With this you should get a `(.venv)` prefix in front of your path in CMD.
```shell
(.venv) $  
```

### Setting up the project directory
- Create a directory named `weights` and a directory named `dataset`.
- Either install the dataset files (CSV) from the above provided links or extract from the `*.rar` files in this repository.
- Make sure the directories are created in the same location as the virtual environment directory `.venv`.
- Also, the project files (Jupyter Notebook files: `*.ipynb`) should remain in the same directory as the `.venv` directory. 

### Download the code files
Download all the Jupyter Notebook files (`*.ipynb`) and setup your kernel in your IDE.
Use the `python.exe` from the virtual environment, install the required libraries and run the notebook. 

## Using the Pre-trained Model for Prediction

- Set-up your virtual environment (as described above)

- Download the weights (from [here](https://github.com/prerakl123/Emotion-Age-Gender-Ethnicity-Prediction/tree/master/modelled/weights)), 
extract and save them in the `weights` directory.
    > There should be 4 `*.h5` files.

- Download the `requirements.txt` and `predict.py` file (from [here](https://github.com/prerakl123/Emotion-Age-Gender-Ethnicity-Prediction/blob/master/modelled/predict.py)), and make sure to install all the libraries from the requirements file.
    ```shell
    (.venv) $ pip install -r requirements.txt 
    ```

- Finally, ensure that the cameras are operational and clean, then run the `predict.py` file:
    ```shell
    (.venv) $ python predict.py
    ```
