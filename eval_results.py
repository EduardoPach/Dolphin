import json
import argparse
from pathlib import Path

import nltk
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

def load_prediction(file_path: Path) -> str:
    df = pd.read_json(file_path)
    return "\n".join(df["text"].tolist())

def load_ground_truth(file_path: Path) -> pd.DataFrame:
    df = pd.read_json(file_path)
    df["conversations"] = df["conversations"].apply(lambda x: x[1]["value"])
    df = df.rename(columns={"conversations": "text"})
    return df

# Same as https://github.com/ucaslcl/Fox/blob/main/eval_tools/eval_ocr_test.py#L35
def compute_edit_distance(pred: str, gt: str) -> float:
    return nltk.edit_distance(pred, gt) / max(len(pred), len(gt))

def main(args: argparse.Namespace) -> None:
    ground_truth_file = Path(args.ground_truth_file)
    prediction_dir = Path(args.prediction_dir)

    # Main columns are: `image` and `text`
    # `image` is the image associated with the converstaion (with .png extension)
    # `text` is the text associated with the image document
    gt_df = load_ground_truth(ground_truth_file)

    # Loop through rows of gt_df and compute Edit Distance between `text` and `prediction`
    edit_distances = []
    
    tqdm.write(f"🔍 Processing {len(gt_df)} documents...")
    tqdm.write("=" * 50)
    
    for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Computing edit distances"):
        prediction_path = prediction_dir / Path(row["image"]).with_suffix(".json")
        if not prediction_path.exists():
            tqdm.write(f"  📄 {Path(row['image']).stem}: Prediction not found")
            continue
        pred_text = load_prediction(prediction_path)
        gt_text = row["text"]
        edit_distance = compute_edit_distance(pred_text, gt_text)
        edit_distances.append((prediction_path.stem, edit_distance))
        
        # Print intermediate results for every 10th document or if edit distance is notably high/low
        if (idx + 1) % 10 == 0 or edit_distance > 0.8 or edit_distance < 0.1:
            tqdm.write(f"  📄 {Path(row['image']).stem}: Edit Distance = {edit_distance:.4f}")

    # Compute statistics and format as a beautiful table
    ed_df = pd.DataFrame(edit_distances, columns=["image", "edit_distance"])
    ed_df.to_csv(f"{prediction_dir}/edit_distances.csv", index=False)
    print(f"Saved edit distances to {prediction_dir}/edit_distances.csv")
    ed_description = ed_df["edit_distance"].describe()
    
    # Convert to table format
    table_data = []
    for stat_name, value in ed_description.items():
        table_data.append([stat_name.title(), f"{value:.6f}"])
    
    # Print beautiful table
    print("\n" + "="*50)
    print("📊 EDIT DISTANCE EVALUATION RESULTS")
    print("="*50)
    print(tabulate(
        table_data,
        headers=["Statistic", "Edit Distance"],
        tablefmt="grid",
        stralign="left",
        numalign="right"
    ))
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-file", type=str, required=True)
    parser.add_argument("--prediction-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)