import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image


# Load CSVs
@st.cache_data
def load_data():
    train_df = pd.read_csv("train_labels.csv")
    val_df = pd.read_csv("val_labels.csv")
    return train_df, val_df


train_df, val_df = load_data()

# Sidebar - Dataset selection
st.sidebar.title("BDD100K Dashboard")
dataset = st.sidebar.selectbox("Select Dataset", ["Train", "Val"])
df = train_df if dataset == "Train" else val_df

# Sidebar - Object category filter
unique_classes = sorted(df["category"].unique())
selected_class = st.sidebar.selectbox(
    "Filter by Class",
    ["All"] + unique_classes,
)

# Filtered Data
if selected_class != "All":
    df = df[df["category"] == selected_class]

# --- Class Distribution Chart ---
st.header(f"{dataset} Set Class Distribution")
class_counts = df["category"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_counts.index, class_counts.values, color="lightseagreen")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Object Count per Class")
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 20,
        f"{height}",
        ha="center",
        va="bottom",
    )
st.pyplot(fig)

# --- Image Viewer ---
st.header("Image Viewer with Bounding Boxes")
master_dir = os.path.join(
    "/nfs/interns/kuldeepk/Assignment/",
    "bdd100k_images_100k/bdd100k/images/100k/",
)
image_dir = (
    os.path.join(master_dir, "train")
    if dataset == "Train"
    else os.path.join(master_dir, "val")
)

sample_images = df["image"].unique()
selected_image = st.selectbox("Select an Image", sample_images)

image_path = os.path.join(image_dir, selected_image)
image = Image.open(image_path)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(image)
sample_boxes = df[df["image"] == selected_image]

for _, row in sample_boxes.iterrows():
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        y1 - 5,
        row["category"],
        color="white",
        fontsize=9,
        backgroundcolor="black",
    )

ax.axis("off")
ax.set_title(f"Annotations for {selected_image}")
st.pyplot(fig)

# --- Train/Val Comparison ---
st.header("Train vs Val Distribution Comparison")
train_counts = train_df["category"].value_counts()
val_counts = val_df["category"].value_counts()
all_classes = sorted(set(train_counts.index).union(val_counts.index))

train_vals = [train_counts.get(c, 0) for c in all_classes]
val_vals = [val_counts.get(c, 0) for c in all_classes]

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(all_classes))
ax.bar(
    x,
    train_vals,
    width=0.4,
    label="Train",
    align="center",
    color="skyblue",
)
ax.bar(
    [i + 0.4 for i in x],
    val_vals,
    width=0.4,
    label="Val",
    align="center",
    color="salmon",
)
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(all_classes, rotation=45)
ax.set_ylabel("Object Count")
ax.set_title("Train vs Val Distribution")
ax.legend()
st.pyplot(fig)
