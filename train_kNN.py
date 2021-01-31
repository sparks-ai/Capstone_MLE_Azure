from sklearn.neighbors import KNeighborsRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '9e65f93e-bdd8-437b-b1e8-0647cd6098f7'
resource_group = 'aml-quickstarts-136648'
workspace_name = 'quick-starts-ws-136648'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='House')
dataset.to_pandas_dataframe()


run = Run.get_context()

def clean_data(data):

    # Cleaning has been done on local computer by external company (so here just some basic cleaning), transform into a dataframes with dependent variable and independent variables
    x_df = data.to_pandas_dataframe()
    y_df = x_df['Transactieprijs_m2']
    x_df.drop('Transactieprijs_m2', inplace = True, axis=1)
    
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default=2, help="Number of neighbors to use")
    parser.add_argument('--leaf_size', type=int, default=20, help="Leaf size passed to algorithm")

    args = parser.parse_args()

    run.log("Neighbors:", np.int(args.n_neighbors))
    run.log("Leafsize:", np.int(args.leaf_size))
    
    x_df, y_df = clean_data(data = dataset)
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.20, random_state = 123)
    
    model = KNeighborsRegressor(n_neighbors=args.n_neighbors, leaf_size = args.leaf_size).fit(x_train, y_train)

    preds = model.predict(x_test) 

    mae = mean_absolute_error(y_test, preds)
          
    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, './outputs/model.joblib')
    
    run.log("mae", np.float(mae))

if __name__ == '__main__':
    main()













import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import argparse
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
x_df = pd.read_csv("E:\\Utrecht_Udacity_MLE_Azure.csv"); print('Klaar met: data inladen')
y_df = x_df['Transactieprijs_m2']
x_df.drop('Transactieprijs_m2', inplace = True, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.20, random_state = 123)

model = KNeighborsRegressor(n_neighbors=5, leaf_size = 20).fit(x_train, y_train)
