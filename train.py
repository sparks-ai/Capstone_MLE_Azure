from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

ds = TabularDatasetFactory.from_delimited_files(path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")

run = Run.get_context()

def clean_data(data):

    # Cleaning has been done on local computer by external company, so transform into a dataframes with dependent variable and independent variables
    x_df = data.to_pandas_dataframe()
    y_df = x_df['Transactieprijs_m2']    
    x_df = x_df.drop('Transactieprijs_m2', axis=1)
    
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=8, help="Maximum depth of trees, where deeper patterns can be found with an increase in depth")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")

    args = parser.parse_args()

    run.log("Max depth:", np.int(args.max_depth))
    run.log("Trees:", np.int(args.n_estimators))
    
    x, y = clean_data(ds)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state = 123)
    
    model = RandomForestRegressor(criterion = "mae", max_depth=args.max_depth, n_estimators = args.n_estimators).fit(x_train, y_train)

    preds = model.predict(x_test) 

    mdape = np.median(np.abs((preds - y_test)/y_test)) * 100
    mape = np.mean(np.abs((preds - y_test)/y_test)) * 100
          
    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, './outputs/model.joblib')
    
    run.log("Median absolute percentage error:", np.float(mdape))
    run.log("Mean absolute percentage error:", np.float(mape))

if __name__ == '__main__':
    main()