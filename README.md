# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Repository structure](#structure)
3. [Instructions](#Instructions)
4. [Results](#results)


## Project Motivation: <a name="motivation"></a>
The aim of this project is to facilitate the work of first line responders during a disaster. When a disaster occurs, many emergency messages are sent out to request food, clothes, medical help etc. However, this large volume of messages also means a lot of work must be done to understand the messages' context and to distinguish the urgent ones from the others.

To make the disaster response faster and more efficient, we will build a web page that can help workers categorize the large quantity of messages left during a disaster response. This will allow people in need to receive the appropriate help they need rapidly.

We develop an ETL and ML pipeline that will use a large training dataset to build a model that can predict the category of any message that is input on the web page.

## Repository structure: <a name="structure"></a>
The structure of the repository is the following:

- /DisasterPipeline/

    - app/
    
            - run.py
            
         - templates/
         
                - go.html
            
            
                - master.html
 
 
    - data/
    
            - DisasterResponse.db
            
            - disaster_categories.csv
            
            - disaster_messaged.csv
            
            - process_data.py


    - models/
    
            - __init__.py
            
            - classifier.pkl
            
            - custom_token_function.py
            
            - train_classifier.py
            
            
    - webapp/
    
            - webpage1.jpg
            
            - webpage2.jpg
            
            - webpage3.jpg
        

## Instructions: <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains the classifier and saves it:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:
    `python run.py`

3. Go to http://0.0.0.0:3000/

4. You will find the produced webpage and you can now input a message in the search bar to correctly classify it.


## Results: <a name="results"></a>
Our webpage can now classify a disaster message that was sent. In addition, two visualizations have been produced to consider the messages' distribution of genres, and the top 5 categories of messages we can expect.

All those visuals can be found in the folder /DisasterPipeline/webapp/.

We do note that our training dataset was very imbalanced with most messages belonging in the 'related' category. This leads to an under-representation of the minority classes in our model. Consequently, some of our messages might be incorrectly classified as 'related' for instance.