# Capstone project Azure Machine Learning Engineer 

## Table of Contents  
[Overview](#overview)  
[Architectural Diagram](#architecture) <br>
[Key Steps](#key_steps) <br> 
[Screen Recording](#recording) <br>
[Future improvements](#future) <br>
<br>   

<a name="overview"/>

## Overview
This project is part of the Udacity Azure ML Nanodegree. The dataset contains data about house prices and characteristics. Based on this data, Azure will be used to first create a scikit-learn KNeighborsRegressor, where two hyperparameters (n_neighbors and leaf_size) will be tuned by using Azure's HyperDrive. Next, the same dataset is served to AutoML to configure a cloud-based machine learning production model, deploy it (because the performance of this model is better than the tuned KNeighborsRegressor model), and finally consume it. Besides, a pipeline will be created, published and consumed. Some of the printscreens are unclear. This is a result of the Virtual Machine used.   
<br>

<a name="architecture"/>

## Architectural Diagram
The steps to take are visualized in the figure below (source: taken from the Udacity Azure Machine Learning Engineer Nanodegree, Capstone project description).
<br>
<br>
![alt text](https://github.com/sparks-ai/Capstone_MLE_Azure/blob/main/Images/capstone-diagram_Udacity.png)
<br>

<a name="key_steps"/>

## Key Steps
The Key steps follow the architectural diagram and provide more details via screenshots. 
<br>

### Authentication
The Azure Machine Learning Extension needs to be installed in order to interact with Azure Machine Learning Studio (part of the az command). Then, a Service Principal account needs to be created an associated with a specific workspace. Afterwards, security and authentication are enabled and completed. 
<br>

### AutoML Experiment
The Bankmarketing dataset is uploaded and a compute cluster (Standard_DS12_v2 with 1 as the number of mininimum nodes) is created in order to configure an AutoML experiment. The AutoML experiment is a classification problem with a yes/no response. The exit criteria are such that the default 3 hours are set to 1 and the concurrency is set from default ot 5. The reason for this is that the focus is on operationalizing machine learning rather than achieving the best possible model outcomes. The screenshots below show that the Bankmarketing dataset is available (one was preloaded and I wanted to upload it myself which is the reason for 2 datasets), the AutoML experiment is completed and the resulting best model.
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/AutoML_Registered_Datasets.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/AutoML_experiment_completed_2.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/AutoML_best_model1.png)
<br>

### Deployment
Now that the experiment is completed and the best model is selected (the VotingEnsemble with a accuracy of 91.65%), it can be deployed into production. The voting ensemble is an ensemble learner that takes an "average" of the other top-performing models. Within Azure Machine Learning Studio, this model can be clicked and deployed with the click on a button. Deploying it will allow to interact with the HTTP API service and interact with the model by sending data over POST requests. Within the deployment settings, authentication is enabled and the model is deployed by means of an Azure Container Instance (ACI). The screenshot below shows the deployment settings for the model.    
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Deployment.png)
<br>

### Logging
In order to retrieve logs, Application Insights has been enabled. This can be done in two ways: either as part of the deploy settings (checkbox) or afterwards via the terminal. For the last option, the az needs to be installed as well as the Python SDK for Azure. The below screenshots show that Application Insights are enabled and it shows the output of logs (a Python script). 
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/ApplicationInsights_enabled.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/LogsPY.png)
<br>

### Documentation
Swagger documentation helps in explaining what the deployed model needs as input in order to deliver a valid response. For the deployed model, a Swagger JSON file can be downloaded from within Azure (within the Endpoints section by clicking on the deployed model). There are two important parts of the Swagger documentation: a swagger.sh file that downloads the latest Swagger container and a serve.py file that will start a Python server. All three files need to be in the same directory. When Swagger runs on localhost, it can be accessed via the webbrowser. The screenshots below show that Swagger is running on localhost. 
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Swagger_contents1.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Swagger_contents2.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Swagger_contents3.png)
<br>

### Consuming model endpoints
When the model is deployed, the scoring_uri and the key can be copied from Azure and pasted in the endpoint.py file. The endpoint.py file contains a JSON payload (2 cases where all the independent variables are provided and which can be used to derive inference on). The endpoint.py file can be executed in the terminal. The first screenshot below shows the endpoint.py script that is running against the API and producing JSON output (data.json file in the directory). In order to load-test the model, the Apache Benchmark commandline tool is installed. The benchmark.sh file is run and the the result of that is provided in the second screenshot below.  
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/EndpointPY_results.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/ApacheBenchmark.png)
<br>

### Create, Publish and Consume a Pipeline
Apart from mainly clicking around in Azure Machine Learning Studio, we can programmatically achieve the same by means of using the Python SDK. A Jupyter notebook is provided which is adjusted here and there. This notebook is available in the directory with the name "aml-pipelines-with-automated-machine-learning-step-total2.ipynb". Please have a look to see the documentation and the resulting output. The steps in this notebook are: creating an experiment in an existing workspace, creating or attaching an existing AmlCompute to a workspace, defining data loading in a TabularDataset, configuring AutoML using AutoMLConfig, using AutoMLStep, training the model using AmlCompute, exploring the results and testing the best fitted model. The screenshots below show that the pipeline has been created, there is a pipeline endpoint, the Bankmarketing dataset with the AutoML module is there (third one added to the two already there), the REST endpoint and a status of active, the Jupyter Notebook widget output and the ML studio showing the scheduled run.  
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Extra_pipeline_running.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Pipeline_created.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Bankmarketing_Dataset_AutoML_module.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Pipeline_endpoint.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Pipeline_endpoint2.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Extra_Pipeline_MLOps_project.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Extra_Pipeline_completed.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/AutoML_experiment_completed_2.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Widget_1.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Best_model.png)
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Performance_model.png)
<br>

<a name="recording"/>

## Screen Recording
Further documentation is in the form of a short screencast. Please go to https://vimeo.com/487342635 to see a screencast of the provided work on Azure. 
<br>

<a name="future"/>

## Future Improvements
This project can be improved in a number of ways. First, a pipeline has been created to run an AutoML experiment. Several other pipelines can be constructed, e.g. that trigger retraining when new data is there or pipelines that perform data cleaning and feature engineering operations. Also, a batch inference pipeline can be constructed in order to derive inference on large samples. In addition, other metrics can be used to optimize the machine learning models. The response variable is imbalanced (see screenshot below). Sampling could be an option or using e.g. the precision-recall AUC as evaluation metric. Also running the AutoML experiment for longer and with different parameters might result in a better model.  
<br>
<br>
![alt text](https://github.com/sparks-ai/MLOps_Azure/blob/master/Images/Standout_class_imbalance.png)
<br>
