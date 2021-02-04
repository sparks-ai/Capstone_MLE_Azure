# Capstone project Azure Machine Learning Engineer 

## Table of Contents  
[Dataset and overview](#overview)  
[Scikit-learn and hyperparameter tuning](#hyperdrive) <br> 
[AutoML steps by manually clicking around](#key_steps) <br> 
[AutoML steps by using the Python SDK](#AutoML_SDK) <br> 
[Screen Recording](#recording) <br>
[Future improvements](#future) <br>
[Standout Suggestions](#standout) <br>
<br>   

<a name="overview"/>

## Dataset and overview
This project is part of the Udacity Azure ML Nanodegree. Predicting house prices using machine learning algorithms is a relevant topic. A dataset with transactional details, housing characteristics and locational characteristics for the city of Utrecht, the Netherlands has been provided by an external company. There are 51,314 rows and 80 columns (including target column). After preprocessing and feature reduction (done on a local computer), the dataset has been imported to the Azure ecosystem, after which models have been trained by using a KNeighborsRegressor scikit-learn model which is tuned by using Azure's HyperDrive and Automated ML. Model performance is evaluated and the best model is deployed and compared against performance benchmarks given by actual valuers. Besides, a pipeline will be created, published and consumed. Some of the printscreens are unclear. This is a result of the Virtual Machine used.   
<br>

### Task
The task is to explain house prices by using building and locational characteristics. In literature, this is called hedonic regression. The dependent variable is the price and the other variables are features. 
<br>

### Access
The dataset has been made available by a third party as .csv file. They've sent this data to me by WeTransfer. In order to get the data in the Azure ecosystem, I downloaded the data from the WeTransfer link on the computer in the workspace. I've put the data on the desktop. In order to get the data into Azure, I went to the Azure Portal and within the Machine Learning studio I went to the dataset tab. There, I could import the data from local files. I've browsed to the location on the desktop and imported the data. The encoding was UTF-8 and the first line contained the headers. Azure puts the data into a blob storage. From within a Python Notebook, we can import data by first setting the workspace by using the subscription_id, resource_group and workspace_name. The dataset can then be imported with the function Dataset.get_by_name and using the workspace and the name of the dataset. The dataset is imported as a tabular dataset. In the Python Notebook, we can do analysis by converting it to a pandas dataframe. However, the AutoML model used the Tabular dataset.    
<br>

### Architectural Diagram
The steps to take are visualized in the figure below (source: taken from the Udacity Azure Machine Learning Engineer Nanodegree, Capstone project description).
<br>
<br>
![Diagram](Images/capstone-diagram_Udacity.png)
<br>

<a name="hyperdrive"/>

## Scikit-learn and hyperparameter tuning
First, I chose to train a KNeighborsRegressor model by using Scikit-learn. The reason I chose for the KNeighborsRegressor is that within real estate valuation, professional valuers often look for the top-X "best comparables" in the neighborhood. This algorithm mimicks this behavior, which is why I thought this would be a good benchmark. Also, the model is intuitively simple and therefore explainable. The two most important hyperparamters that I tuned for the algorithm are n_neighbors and leaf_size. The amount of neighbors relates to how many comparables in the direct neighborhood should be evaluated. This is where I know professional valuers often pick their top-3 or top-5. This parameter could take values 2, 5, and 10. Then the leaf_size. This determines the complexity and the amount of variables used. Given that professional valuers look at location and basic building characteristics, I chose ranges for this hyperparameter between 15 and 50. Location in itself can of course be decomposed into numerous variables in the dataset.  
<br>

### Results
The best resulting KNeighborsRegressor model had as hyperparameters a value for n_neighbors of 2 and for leaf_size of 42. This is interesting as it means being very local (given the amount for n_neighbors) but achieving complexity on this local level by using many variables. In literature, the mean average error (MAE) is often used as a metric for evaluating results of house price prediciton algorithms. For this model, the MAE is 525.55. 
This model could potentially be improved by broadening the hyperparameter space (e.g. n_neighbors between 1 and 10 and leaf_size between 15 and 80). Also, there are more hyperparameters than could have been tuned, e.g. the algorithm (auto vs. ball_tree vs kd_tree vs brute), the power parameter for the Minkowski metric and the distance metric to use.  
<br>
<br>
The screenshots below provide more insights into the model and its results. These can also be found in the notebook in the directory. 
<br>
<br>
Model registered:
<br>
![Model_registered](Images/Hyperdrive_model_registered.png)
<br>
Completed run:
<br>
![Completed_run](Images/Hyperdrive_widget_1.png)
<br>
Best metrics:
<br>
![Best_metrics](Images/Hyperdrive_widget_2.png)
<br>
End results:
<br>
![End_results](Images/Hyperdrive_outcomes.png)
<br>
Model registered: 
<br>
![Model_registered2](Images/Hyperdrive_best_model_run_id_registered.png)
<br>

<a name="key_steps"/>

## AutoML steps by manually clicking around
This section will highlight all steps for the AutoML model, as well as for publishing and pipeline endpoint (as standout suggestion).
<br>

### Authentication
The Azure Machine Learning Extension needs to be installed in order to interact with Azure Machine Learning Studio (part of the az command). Then, a Service Principal account needs to be created an associated with a specific workspace. Afterwards, security and authentication are enabled and completed. 
<br>

### AutoML Experiment
The dataset is uploaded and a compute cluster is created in order to configure an AutoML experiment. The AutoML experiment is a regression problem with a numeric response. The exit criteria are such that the default 3 hours are left as is and the concurrency is set from default to 5. The reason for this is that the dataset is relatively small, but I want to be sure to get the best possible model. The screenshots below show that the dataset is available, the AutoML experiment is completed and the resulting best model.
<br>
<br>
![Dataset_active](Images/Dataset_active.png)
<br>
![Best_mod1](Images/Best_model1.png)
<br>
![Best_mod2](Images/Best_model2.png)
<br>
![Best_mod3](Images/Best_model3.png)
<br>

### Results
The best resulting model is a VotingEnsemble, which is a combination of the other top-performing algorithms. The MAE for this ensemble is 292.83, which is substantially lower than the MAE of 525.55 for the tuned KNeighborsRegressor model.  
This model could potentially be improved by including deep learning models as well within AutoML and by adding extra input features to the dataset because quite complex algorithms are on top. 

### Deployment
Now that the experiment is completed and the best model is selected (the VotingEnsemble), it can be deployed into production. The voting ensemble is an ensemble learner that takes an "average" of the other top-performing models. Within Azure Machine Learning Studio, this model can be clicked and deployed with the click on a button. Deploying it will allow to interact with the HTTP API service and interact with the model by sending data over POST requests. Within the deployment settings, authentication is enabled and the model is deployed by means of an Azure Container Instance (ACI). The screenshot below shows the deployment settings for the model.    
<br>
<br>
![Endpoint_model](Images/Endpoint_model.png)
<br>

### Logging
In order to retrieve logs, Application Insights has been enabled. This can be done in two ways: either as part of the deploy settings (checkbox) or afterwards via the terminal. For the last option, the az needs to be installed as well as the Python SDK for Azure. The below screenshots show that Application Insights are enabled. 
<br>
<br>
![Logging](Images/Logging.png)
<br>

### Documentation
Swagger documentation helps in explaining what the deployed model needs as input in order to deliver a valid response. For the deployed model, a Swagger JSON file can be downloaded from within Azure (within the Endpoints section by clicking on the deployed model). There are two important parts of the Swagger documentation: a swagger.sh file that downloads the latest Swagger container and a serve.py file that will start a Python server. All three files need to be in the same directory. When Swagger runs on localhost, it can be accessed via the webbrowser. The screenshots below show that Swagger is running on localhost. 
<br>
<br>
![Swagger1](Images/Swagger1.png)
<br>
![Swagger2](Images/Swagger2.png)
<br>

### Consuming model endpoints
When the model is deployed, the scoring_uri and the key can be copied from Azure and pasted in the endpoint.py file. The endpoint.py file contains a JSON payload (2 cases where all the independent variables are provided and which can be used to derive inference on). The endpoint.py file can be executed in the terminal. The screenshots below shows the endpoint.py script that is running against the API and producing JSON output (data.json file in the directory).  
<br>
<br>
![Rest](Images/REST_endpoint_URL.png)
<br>
![Data_to_model](Images/Data_sent_to_model_endpoint.png)
<br>

### Create, Publish and Consume a Pipeline
Apart from mainly clicking around in Azure Machine Learning Studio, we can programmatically achieve the same by means of using the Python SDK. A Jupyter notebook is provided which is adjusted here and there. This notebook is available in the directory. Please have a look to see the documentation and the resulting output. The steps in this notebook are: creating an experiment in an existing workspace, creating or attaching an existing AmlCompute to a workspace, defining data loading in a TabularDataset, configuring AutoML using AutoMLConfig, using AutoMLStep, training the model using AmlCompute, exploring the results and testing the best fitted model. The screenshots below show that the pipeline has been created, there is a pipeline endpoint, the REST endpoint and a status of active, the Jupyter Notebook widget output and the ML studio showing the scheduled run.  
<br>
<br>
![Exp](Images/Pipeline_experiment.png)
<br>
![Exp_completed](Images/Pipeline_experiment_completed.png)
<br>
![Exp_completed2](Images/Pipeline_experiment_completed2.png)
<br>
![Pipeline_widget](Images/Pipeline_widget.png)
<br>
![Pipeline_best_model](Images/Pipeline_best_model.png)
<br>
![Pipeline_best_model2](Images/Pipeline_best_model2.png)
<br>
![Pipeline_best_model3](Images/Pipeline_best_model3.png)
<br>
![Pipeline_runs](Images/Pipeline_runs.png)
<br>
![Pipeline_published](Images/Pipeline_published.png)
<br>

<a name="AutoML_SDK"/>

## AutoML steps by using the Python SDK
The same AutoML model can be derived by using code in the Python SDK. In this section, the steps to derive the AutoML model by means of code will be discussed.   
<br>
In the directory, there's a Python Notebook called AutoML_SDK, which contains the steps to run an AutoML experiment with configered settings, retrieve the best model and register this model. The notebook is not documented, as it is very similar to the other notebook with the AutoML pipeline. The AutoMLConfig file is constructed in such a way that it uses the created compute_target, the task is set to regression, the dataset contains the dataset and the validation set size is similar to the settings of the scikit-learn HyperDrive run (20%). Early stopping is enabled in order to stop if results on the validation set show no further improvements. The intention of this is to reduce training time, which comes in handy when many models are trained, as is the case with AutoML. Featurization is set to auto, which means that within the AutoML run, automated feature engineering is performed. This can increase model performance. Logs are written to a file in order to more easily debug if errors arise. In the automl_settings, a timeout in minutes of the whole experiment is set to 30 (as this example is just to show that AutoML can be performed via the SDK) and the max_concurrent_iterations is set to 5. This means that 5 models can be trained in parallel. The primary metric to optimize for the AutoML models is the spearman_correlation. This is the default metric for regression problems. I first thought to change this to the MAE, as this metric is compared between models. But later I changed it back to spearman_correlation and found a better overall performance (also a slightly better MAE for the resulting model).     
<br>

### Results
The best resulting VotingEnsemble model has an MAE of 295.76. The amount of estimators is 25, the objective is to optimize the squarederror and other hyperparameters are given in the screenshots below. This VotingEnsemble combines the top performing models and we can see the weights of models. The best performing model in the VotingEnsemble is a LightGBMRegressor, which uses 100 estimators (and more hyperparameters as in the printscreens below). This model has been registered and is now available in the models section of Azure ML Studio. 
<br>
<br>
The screenshots below provide more insights into the model and its results. These can also be partly found in the notebook in the directory. 
<br>
<br>
![Run](Images/AutoML_run.png)
<br>
![Mod1](Images/AutoML_run_best_model1.png)
<br>
![Mod2](Images/AutoML_run_best_model2.png)
<br>
![Config](Images/AutoML_config.png)
<br>
![Retrieving_best_model](Images/Retrieving_best_model.png)
<br>
![Best](Images/Best_model.png)
<br>
![Final_estimator](Images/Retrieving_best_model_final_estimator.png)
<br>
![Final_estimator](Images/Best_model_registered.png)
<br>

<a name="recording"/>

## Screen Recording
Further documentation is in the form of a short screencast. Please go to https://vimeo.com/508176649 to see a screencast of the provided work on Azure. 
<br>

<a name="future"/>

## Future Improvements
This project can be improved in a number of ways. First, a pipeline has been created to run an AutoML experiment. Several other pipelines can be constructed, e.g. that trigger retraining when new data is there or pipelines that perform data cleaning and feature engineering operations. This makes everything better in terms of efficiency (automation). In such a way, automated retraining is performed which makes the solution "hands off". In addition, a data cleaning pipeline could ensure that improper data entry does not influence model performance too much. Much of the input data should not be asked to a client and is data that can be easily merged. For that, enriching basic inputs given by a client by means of merging with a database in the backend could be done. A pipeline for this can be created. This will make the total solution much more client-friendly. Also, a batch inference pipeline can be constructed in order to derive inference on large samples. As of now, a json input is expected and this has been tested with a json payload of 2 samples. For large samples, a batch inference pipeline ensures that results are delivered on time (automated resizing). Also running the AutoML experiment for longer and with different parameters might result in a (slightly) more accurate model. When accuracy is key, waiting for more iterations can be beneficial. Next to that, collecting more data in terms of length (amount of records) and width (amount of variables) could improve accuracy as well. Continuously adding more data in terms of records and different variables is the manual task that remains. Time should be invested in this task whereas much of the machine learning tasks have been automated.    
<br>

<a name="standout"/>

## Standout Suggestions
Next to deploying a model, I've also deployed a pipeline. Also, I implemented logging for the model. 
