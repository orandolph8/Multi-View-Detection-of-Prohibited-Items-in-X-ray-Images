import json
import os
import shutil

# Mapping dictionary for translating Chinese class names to English
CN_TO_EN_MAP = {
    '手机': 'phone',
    '橙色液体': 'orange_liquid',
    '不带电芯充电宝': 'non_battery_charger',  # Non-battery power bank
    '电脑': 'laptop',
    '绿色液体': 'green_liquid',
    '带电芯充电宝': 'battery_charger',         # Battery power bank
    '平板电脑': 'tablet',
    '蓝色液体': 'blue_liquid',
    '柱状橙色液体': 'cylindrical_orange_liquid',
    '非金属打火机': 'non_metal_lighter',
    '雨伞': 'umbrella',
    '柱状绿色液体': 'cylindrical_green_liquid',
}


def convert_coco_to_yolo(json_path, image_root_dir, output_dir, data_type="train_A"):
    """
    Converts COCO-format annotations (like LDXray) to YOLOv8 structure:
      dataset/train_A/images/
      dataset/train_A/labels/

    Args:
        json_path (str): Path to COCO annotation file (e.g., 'LDXray/train.json')
        image_root_dir (str): Directory containing images (e.g., 'LDXray/dataset/train_A')
        output_dir (str): Output dataset split directory (e.g., 'LDXray/dataset/train_A')
        data_type (str): Split name (e.g., 'train_A', 'val_A')
    """
    print(f"\n--- Starting conversion for {data_type} ---")
    print(f"Loading annotations from: {json_path}")

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # --- Load JSON ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Build category mapping ---
    categories = sorted(data["categories"], key=lambda x: x["id"])
    category_map = {cat["id"]: idx for idx, cat in enumerate(categories)}
    yolo_class_names = [CN_TO_EN_MAP.get(cat["name"], cat["name"]) for cat in categories]

    print(f"Detected {len(yolo_class_names)} classes:")
    for i, name in enumerate(yolo_class_names):
        print(f"  {i}: {name}")

    # Save class names file (once, at dataset root)
    class_file = os.path.join(os.path.dirname(output_dir), "class_names.txt")
    with open(class_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_class_names))
    print(f"Saved class names to: {class_file}")

    # --- Build image lookup ---
    image_lookup = {img["id"]: img for img in data["images"]}
    label_count = 0
    missing_images = 0

    # --- Convert annotations ---
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_lookup:
            missing_images += 1
            continue

        img_info = image_lookup[img_id]
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

        # Normalize bbox (COCO -> YOLO)
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        cat_id = ann["category_id"]
        yolo_cat_id = category_map[cat_id]

        # Label path
        label_file = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(label_file, "a", encoding="utf-8") as f:
            f.write(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        label_count += 1

        # Copy image if not yet in /images/
        src_img_path = os.path.join(image_root_dir, file_name)
        dst_img_path = os.path.join(images_dir, file_name)
        if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)

    print(f"✅ Converted {label_count} annotations for {len(image_lookup)} images.")
    if missing_images > 0:
        print(f"⚠️ Skipped {missing_images} annotations (missing image info).")
    print(f"YOLO labels saved in: {labels_dir}")


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    LDXRAY_DIR = "LDXray"
    DATASET_DIR = os.path.join(LDXRAY_DIR, "dataset")

    # --- Convert train_A ---
    TRAIN_JSON = os.path.join(LDXRAY_DIR, "train.json")
    TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_A")  # your current image folder
    TRAIN_OUT = os.path.join(DATASET_DIR, "train_A")
    convert_coco_to_yolo(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_OUT, data_type="train_A")

    # --- Convert val_A ---
    VAL_JSON = os.path.join(LDXRAY_DIR, "train.json")
    VAL_IMG_DIR = os.path.join(DATASET_DIR, "val_A")
    VAL_OUT = os.path.join(DATASET_DIR, "val_A")

    if os.path.exists(VAL_JSON):
        convert_coco_to_yolo(VAL_JSON, VAL_IMG_DIR, VAL_OUT, data_type="val_A")
    else:
        print(f"⚠️ Skipping val_A conversion — file not found: {VAL_JSON}")

    print("\n🎯 Conversion complete! Your dataset now matches YOLOv8 expected format.")
