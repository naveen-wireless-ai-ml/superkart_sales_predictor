# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download

# Log in to Hugging Face Hub
login(token=os.getenv("HF_TOKEN"))

# Download SuperKart.csv locally first
data_filename = "SuperKart.csv"
data_repo_id = "naveenaggarwal1989/superkart-sales-data"
local_data_path = hf_hub_download(repo_id=data_repo_id, filename=data_filename, repo_type="dataset")

df = pd.read_csv(local_data_path)
print("Dataset loaded successfully.")

# Drop the unique identifier and any unnamed columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
if 'Product_Id' in df.columns:
    df.drop(columns=['Product_Id'], inplace=True)
if 'Store_Id' in df.columns:
    df.drop(columns=['Store_Id'], inplace=True)

# Map 'reg' to 'Regular' in 'Product_Sugar_Content' as it looks like a typo only
if 'Product_Sugar_Content' in df.columns:
    df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace('reg', 'Regular')

target_col = 'Product_Store_Sales_Total'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=10
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="naveenaggarwal1989/superkart-sales-data",
        repo_type="dataset",
    )
