from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "naveenaggarwal1989/SuperKartSalesPredictor"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, space_sdk="docker")
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="src_code/deployment",     # the local folder containing your files
    repo_id=repo_id,                            # the target repo
    repo_type=repo_type,                        # dataset, model, or space
    path_in_repo="",                            # optional: subfolder path inside the repo
)
