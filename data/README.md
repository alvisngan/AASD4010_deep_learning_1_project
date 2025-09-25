# Data Scraper

I am not uploading audio data directly to github, but here is the tools to download the data yourself.

## Tutorial

### Creating the Python Environment

(Note: you will also need ffmpeg installed on your system.)

```sh
# Create python environment
python3 -m venv .venv

# Initiate environment (Unix-like OS's)
source .venv/bin/activate

# Install required packages
pip install -r requirements_yt_sraper.txt
```

### Using [`yt_scraper.py`](./yt_scraper.py)

```sh
# Inside your python environment
# The data will be downloaded in AASD4010_deep_learning_1_project/data/data_dir
python yt_scraper.py --xlsx ./Dataset.xlsx --out data_dir --overwrite
```
