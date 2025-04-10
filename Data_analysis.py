import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def parse_bdd100k_labels(json_path, output_csv=None):
    """
    Parses BDD100K detection labels JSON and extracts relevant data.

    Args:
        json_path (str): Path to the label JSON file.
        output_csv (str, optional): Path to save parsed data as CSV.

    Returns:
        pd.DataFrame: Parsed annotations.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    valid_classes = {
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motorcycle",
        "train",
        "traffic light",
        "traffic sign",
    }

    for item in tqdm(data, desc="Parsing BDD100K labels"):
        image_name = item["name"]
        for label in item.get("labels", []):
            category = label.get("category")
            if category in valid_classes and "box2d" in label:
                box = label["box2d"]
                records.append(
                    {
                        "image": image_name,
                        "category": category,
                        "x1": box["x1"],
                        "y1": box["y1"],
                        "x2": box["x2"],
                        "y2": box["y2"],
                    }
                )

    df = pd.DataFrame(records)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved parsed data to {output_csv}")

    return df


def plot_class_distribution(df, title="Class Distribution", figsize=(10, 6)):
    class_counts = df["category"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=figsize)
    bars = plt.bar(class_counts.index, class_counts.values, color="skyblue")
    plt.xlabel("Object Category")
    plt.ylabel("Number of Instances")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 100,
            str(height),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(f"{title}.png", transparent=False)
    plt.show()


def compare_train_val_distributions(train_df, val_df):
    train_counts = train_df["category"].value_counts().sort_index()
    val_counts = val_df["category"].value_counts().sort_index()

    categories = sorted(set(train_counts.index).union(val_counts.index))

    train_vals = [train_counts.get(cat, 0) for cat in categories]
    val_vals = [val_counts.get(cat, 0) for cat in categories]

    # Plot
    x = range(len(categories))
    plt.figure(figsize=(12, 6))
    plt.bar(
        x,
        train_vals,
        width=0.4,
        label="Train",
        align="center",
        color="skyblue",
    )
    plt.bar(
        [i + 0.4 for i in x],
        val_vals,
        width=0.4,
        label="Val",
        align="center",
        color="salmon",
    )
    plt.xticks([i + 0.2 for i in x], categories, rotation=45)
    plt.ylabel("Number of Instances")
    plt.title("Train vs Val Distribution")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    # Print imbalance ratios
    print("\n--- Class Ratios (Train / Val) ---")
    for cat in categories:
        train_count = train_counts.get(cat, 0)
        val_count = val_counts.get(cat, 1)  # Avoid division by zero
        ratio = round(train_count / val_count, 2)
        print(f"{cat:<15}: {train_count} / {val_count} → Ratio: {ratio}")


# Run it


def main():
    master_dir = os.path.join(
        "/data/",
        "bdd100k_labels_release",
        "bdd100k/labels/",
    )
    train_json = os.path.join(master_dir, "bdd100k_labels_images_train.json")
    val_json = os.path.join(master_dir, "bdd100k_labels_images_val.json")

    train_df = parse_bdd100k_labels(train_json, output_csv="train_labels.csv")
    val_df = parse_bdd100k_labels(val_json, output_csv="val_labels.csv")

    train_df = pd.read_csv("train_labels.csv")
    val_df = pd.read_csv("val_labels.csv")

    # Plot distributions
    plot_class_distribution(train_df, title="Train Set Class Distribution")
    plot_class_distribution(val_df, title="Validation Set Class Distribution")

    compare_train_val_distributions(train_df, val_df)


if __name__ == "__main__":
    main()
