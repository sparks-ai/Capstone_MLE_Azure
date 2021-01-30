import catboost as cb
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import gc
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

ds = TabularDatasetFactory.from_delimited_files(path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/Utrecht_Udacity_MLE_Azure.csv")

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

    parser.add_argument('--max_depth', type=int, default=10, help="Maximum depth of catboost trees. The higher, the more complex relationships will be found")
    parser.add_argument('--iterations', type=int, default=400, help="Maximum number of iterations to converge")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate; order of magnitude of each step to take to go to a local/global minimum")

    args = parser.parse_args()

    run.log("Max depth:", np.int(args.max_depth))
    run.log("Max iterations:", np.int(args.iterations))
    run.log("Learning rate:", np.float(args.learning_rate))
    
    x, y = clean_data(ds)
    
    categorical_features_indices = np.where(x.dtypes != np.float)[0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state = 123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.15, random_state = 4)

    pool = cb.Pool(data =x_train, label=y_train, cat_features=categorical_features_indices)
    del x_train
    del y_train
    gc.collect()

    model = cb.CatBoostRegressor(eval_metric = "MAPE", verbose = True, od_type = "Iter", od_wait = 15, random_seed = 123, iterations = args.iterations, learning_rate = args.learning_rate, l2_leaf_reg = 14, max_depth = args.max_depth, colsample_bylevel = 1, border_count = 128, max_ctr_complexity = 2).fit(pool, eval_set = (x_val, y_val))

    preds = model.predict(x_test) 

    mdape = np.median(np.abs((preds - y_test)/y_test)) * 100
    mape = np.mean(np.abs((preds - y_test)/y_test)) * 100
        
    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, './outputs/model.joblib')
    
    run.log("Median Absolute Percentage Error:", np.float(mdape))
    run.log("Mean Absolute Percentage Error:", np.float(mape))

if __name__ == '__main__':
    main()