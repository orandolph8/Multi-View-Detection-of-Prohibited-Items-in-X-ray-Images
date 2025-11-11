import os
import random
import shutil

# Your dataset structure
train_img_dir = 'LDXray/dataset/train_A/images'
train_lbl_dir = 'LDXray/dataset/train_A/labels'

val_img_dir = 'LDXray/dataset/val_A/images'
val_lbl_dir = 'LDXray/dataset/val_A/labels'

# Create val folders if not exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# List all training images
images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
val_count = int(0.1 * len(images))  # 10% split
val_samples = random.sample(images, val_count)

missing_labels = 0

for img_file in val_samples:
    base = os.path.splitext(img_file)[0]
    label_file = base + '.txt'

    img_src = os.path.join(train_img_dir, img_file)
    lbl_src = os.path.join(train_lbl_dir, label_file)

    img_dst = os.path.join(val_img_dir, img_file)
    lbl_dst = os.path.join(val_lbl_dir, label_file)

    # Move image
    shutil.move(img_src, img_dst)

    # Move label (if exists)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)
    else:
        missing_labels += 1
        print(f"⚠️ No label for image: {img_file}")

print(f"\n✅ Finished splitting dataset!")
print(f"Moved {val_count} images to validation folder.")
print(f"⚠️ Missing labels for {missing_labels} images.")
