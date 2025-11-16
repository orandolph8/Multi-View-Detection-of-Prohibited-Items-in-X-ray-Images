import torch
import cv2
import numpy as np
from model_ResNet import AHCR
from utils import confidence_weighted_view_fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = (224, 224)

classes = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors', 'Lighter',
           'Battery', 'Bat', 'Razor_blade', 'Saw_blade', 'Fireworks',
           'Hammer', 'Screwdriver', 'Dart', 'Pressure_vessel']


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32) / 255.

    # Convert 1-channel → 3-channel for ResNet50
    img = np.stack([img, img, img], axis=0)   # shape: (3, 224, 224)

    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def predict(checkpoint_path, image_id):
    print("Loading checkpoint:", checkpoint_path)

    # allow AHCR class during loading
    torch.serialization.add_safe_globals([AHCR])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = checkpoint["model"].to(device)
    model.eval()

    # ---- Load images ----
    ol_path = f"./data/DvXray/{image_id}_OL.png"
    sd_path = f"./data/DvXray/{image_id}_SD.png"

    img_ol = load_img(ol_path).to(device)
    img_sd = load_img(sd_path).to(device)

    with torch.no_grad():
        ol_out, sd_out = model(img_ol, img_sd)

    ol_prob = torch.sigmoid(ol_out)
    sd_prob = torch.sigmoid(sd_out)

    fused = confidence_weighted_view_fusion(ol_prob, sd_prob)
    fused = fused.squeeze().cpu().numpy()

    print("\n===== Prediction =====")
    for cls, score in zip(classes, fused):
        print(f"{cls:15s}: {score:.4f}")

    print("\nDetected:")
    for cls, score in zip(classes, fused):
        if score > 0.5:
            print(f"✔ {cls}: {score:.3f}")


if __name__ == "__main__":
    predict("checkpoint/ep030_ResNet_checkpoint.pth.tar", "P00000")
