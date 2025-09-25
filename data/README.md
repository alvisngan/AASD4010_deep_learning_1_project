# Data Scraper

We do not upload audio data directly to github, but we provide you with the tools to do so.

## Tutorial

### Creating the Python Environment

(Note: you will also need ffmpeg installed on your system.)

```sh
# Create python environment
python3 -m venv .venv

# Activate environment (Unix-likes)
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
