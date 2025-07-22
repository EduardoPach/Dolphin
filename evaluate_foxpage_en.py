import os
import zipfile
from pathlib import Path
import subprocess

import nltk
import rich
import gdown
import jiwer
import pandas as pd
from huggingface_hub import hf_hub_download

FILE_ID = "1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q"
URL_TEMPLATE = "https://drive.google.com/uc?id={file_id}"
ZIP_PATH = "dataset/foxpage_en.zip"
UNZIP_DIR = "dataset"
HF_MODEL_DIR = "hf_model"
REPO_ID = "ByteDance/Dolphin"

FOX_PAGE_EN_DIR = Path("dataset/Fox-Page-Benchmark/en")
RESULT_DIR = Path("result-fox-page-en")
EVALUATION_RESULTS_PATH = Path("datasets/evaluation_results.csv")

############## Download Data ###############
# After finishing the download an unzip you should have a directory structure like this:
# dataset
# └── Fox-Page-Benchmark
#     ├── en
#     │   ├── gt
#     │   └── img
#     └── zh
#         ├── gt
#         └── img

# 8 directories


def download_data() -> None:
    url = URL_TEMPLATE.format(file_id=FILE_ID)
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    gdown.download(url, output=ZIP_PATH, quiet=False)

def unzip_data() -> None:
    # Create the dataset directory if it doesn't exist
    os.makedirs(UNZIP_DIR, exist_ok=True)

    # Open the zip file and extract all contents into the dataset directory
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(UNZIP_DIR)


############## Loads Ground Truth ###############

def load_one_file(json_path: Path) -> pd.DataFrame:
    return (
        pd.read_json(json_path, typ="series")
        .to_frame()
        .T
        .rename(columns={"image_name": "image", "page_contents": "text"})
        .assign(image=lambda df: df["image"].str.split(".").str[0])
    )

def load_ground_truth(fox_page_dir: Path) -> pd.DataFrame:
    """`fox_page_dir` is the language-specific directory with `img` and `gt` subdirectories"""
    df = pd.concat(
        [load_one_file(p) for p in (fox_page_dir / "gt").glob("*.json")],
        ignore_index=True
    )
    return df

############## Loads Predictions ###############

def load_prediction_sample(file_path: Path) -> str:
    df = pd.read_json(file_path)
    return "\n".join(df["text"].tolist())

def load_predictions(dir_path: Path) -> pd.DataFrame:
    predictions = []
    for file_path in dir_path.glob("*.json"):
        predictions.append((file_path.stem, load_prediction_sample(file_path)))
    return pd.DataFrame(predictions, columns=["image", "text"])

############## Runs Inference ###############

# Equivalent to `huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model`
def download_hf_model() -> None:
    hf_hub_download(
        repo_id=REPO_ID,
        local_dir=HF_MODEL_DIR,
    )

# Equivalent to `python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs --save_dir ./results`
def run_inference(input_path: str, save_dir: str) -> None:
    subprocess.run(
        [
            "python",
            "demo_page_hf.py",
            "--model_path",
            HF_MODEL_DIR,
            "--input_path",
            input_path,
            "--save_dir",
            save_dir,
        ]
    )

############## Evaluates ###############

def evaluate(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = ground_truth.merge(predictions, on="image", how="left", suffixes=("_gt", "_pred")).dropna()
    df["cer"] = df.apply(lambda row: jiwer.cer(reference=row["text_gt"], hypothesis=row["text_pred"]), axis=1)
    global_cer = jiwer.cer(
        reference=df["text_gt"].tolist(), 
        hypothesis=df["text_pred"].tolist()
    )
    return df, global_cer

def display_results(df: pd.DataFrame, global_cer: float) -> None:
    cer_description = df["cer"].describe()
    
    # Create a rich table for CER statistics
    from rich.table import Table
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    # Create table for CER statistics
    stats_table = Table(title="Character Error Rate (CER) Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")
    
    # Add statistics to table
    stats_table.add_row("Count", f"{cer_description['count']:.0f}")
    stats_table.add_row("Mean", f"{cer_description['mean']:.4f}")
    stats_table.add_row("Std", f"{cer_description['std']:.4f}")
    stats_table.add_row("Min", f"{cer_description['min']:.4f}")
    stats_table.add_row("25%", f"{cer_description['25%']:.4f}")
    stats_table.add_row("50% (Median)", f"{cer_description['50%']:.4f}")
    stats_table.add_row("75%", f"{cer_description['75%']:.4f}")
    stats_table.add_row("Max", f"{cer_description['max']:.4f}")
    
    # Create table for global results
    global_table = Table(title="Global Evaluation Results", show_header=True, header_style="bold blue")
    global_table.add_column("Metric", style="cyan", no_wrap=True)
    global_table.add_column("Value", style="green")
    
    global_table.add_row("Global CER", f"{global_cer:.4f}")
    global_table.add_row("Total Samples", f"{len(df)}")
    
    # Display tables
    console.print(Panel(global_table, title="[bold blue]Evaluation Summary", border_style="blue"))
    console.print(Panel(stats_table, title="[bold magenta]Detailed Statistics", border_style="magenta"))
    
    # Display some sample results
    if len(df) > 0:
        sample_table = Table(title="Sample Results (First 5)", show_header=True, header_style="bold yellow")
        sample_table.add_column("Image", style="cyan", no_wrap=True)
        sample_table.add_column("CER", style="green")
        sample_table.add_column("Ground Truth Length", style="blue")
        sample_table.add_column("Prediction Length", style="blue")
        
        for _, row in df.head().iterrows():
            gt_length = len(row["text_gt"])
            pred_length = len(row["text_pred"])
            sample_table.add_row(
                row["image"],
                f"{row['cer']:.4f}",
                str(gt_length),
                str(pred_length)
            )
        
        console.print(Panel(sample_table, title="[bold yellow]Sample Results", border_style="yellow"))


def main():
    # Prepare Data
    download_data()
    unzip_data()
    # Download Model
    download_hf_model()
    # Run Inference
    run_inference(input_path=str(FOX_PAGE_EN_DIR / "img"), save_dir=str(RESULT_DIR))
    # Load Predictions
    predictions = load_predictions(RESULT_DIR / "recognition_json")
    # Load Ground Truth
    ground_truth = load_ground_truth(FOX_PAGE_EN_DIR)
    # Evaluate
    df, global_cer = evaluate(predictions, ground_truth)
    # Save Results
    df.to_csv(EVALUATION_RESULTS_PATH, index=False)
    # Display Results
    display_results(df, global_cer)

if __name__ == "__main__":
    main()
