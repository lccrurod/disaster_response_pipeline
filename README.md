# disaster_response_pipelines

## About the project

This repository is part of the Data Scientist nanodegree from Udacity, the objetive is to, based on data provide by Figure Eight, create a ETL process that feeds the data to a machine learning pipeline that trains and makes available a classifier to predict a message related categories by queriying from a web app.

## Files

### app
#### templates/master.html
Main page to be show in the web app.

#### templates/go.html
Page to be show when the user wants a message to be classified.

#### extra_func.py
Module that contains auxiliar NLP utilities to be used in the app.

#### run.py
Backend code for the web app.

### data
#### disaster_categories.csv
Data containing messages categories and corresponding ids.

#### disaster_messages.csv
Data containing the full message, genre, original message and corresponding ids.

#### messages_application.db
Database containing the processed data from messages and classification results.

#### process_data.py
Code for the ETL process that prepares the data for the ML application.

### models
#### classifier.pkl
Pickle file to store the model state after training process.

#### train_classifier.py
Code for training the message classifier ML model and store the classification metrics.

## Libraries Used

This are available in the **requirements.txt** file

## Execution

### 1. process_data.py

From the data directory run

> `python process_data.py disaster_messages.csv disaster_categories.csv messages_application.db`

### 2. train_classifier.py

From the models directory run

> `python train_classifier.py ../data/messages_application.db classifier.pkl`

### 3. run.py

From the app directory run

>`set FLASK_APP=run`
>
>`flask run`

### 4. The web app should now be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/")

## The web app

![First screenshot!](/assets/screenshot1.png)

![Second screenshot!](/assets/screenshot2.png)


## Acknowlegments

For this particular project I found pretty useful this [thread](https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules, "stackoverflow") about pickling and unpickling objects in python.

When working with GitHub and the pickle ML model was helpful the [guide](https://git-lfs.github.com./,'git-lfs') for working with large size files.

Finally for setting my local CMD for running the flask app and succesfully excecute the web app I refered to the  [quickstart documentation](https://flask.palletsprojects.com/en/2.0.x/quickstart/, "flask-doc") from the Flask team.
