import kagglehub
import os
import shutil

# Path where the dataset is downloaded
# https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share
path = kagglehub.dataset_download("robertsunderhaft/nba-player-season-statistics-with-mvp-win-share")
print("Path to dataset files:", path)

# Create a dataset directory if it doesn't exist
dataset_dir = os.path.join("datasets", "nba-player-season-statistics-with-mvp-win-share")
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)  # Remove the directory if it exists to avoid copytree error
shutil.copytree(path, dataset_dir)

print("Directory copied successfully to:", dataset_dir)
