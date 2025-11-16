import os
import json
import cv2

JSON_DIR = "./data/DvXray"
OUT_OL = "./data/boxed_images/OL"
OUT_SD = "./data/boxed_images/SD"

os.makedirs(OUT_OL, exist_ok=True)
os.makedirs(OUT_SD, exist_ok=True)


def draw_for_one(image_id):
    json_path = f"{JSON_DIR}/{image_id}.json"
    ol_path = f"{JSON_DIR}/{image_id}_OL.png"
    sd_path = f"{JSON_DIR}/{image_id}_SD.png"

    with open(json_path, 'r') as f:
        data = json.load(f)

    img_ol = cv2.imread(ol_path)
    img_sd = cv2.imread(sd_path)

    if img_ol is None or img_sd is None:
        print(f"⚠️ Missing image for {image_id}, skipping...")
        return

    for obj in data["objects"]:
        label = obj["label"]

        # OL box
        ol_bb = obj["ol_bb"]
        if not isinstance(ol_bb, list) or len(ol_bb) != 4:
            print(f"⚠️ Skipping malformed ol_bb in {image_id}: {ol_bb}")
            continue
        x1, y1, x2, y2 = ol_bb
        cv2.rectangle(img_ol, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_ol, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # SD box
        sd_bb = obj["sd_bb"]
        if not isinstance(sd_bb, list) or len(sd_bb) != 4:
            print(f"⚠️ Skipping malformed sd_bb in {image_id}: {sd_bb}")
            continue
        x1, y1, x2, y2 = sd_bb
        cv2.rectangle(img_sd, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_sd, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(f"{OUT_OL}/{image_id}_OL_boxed.png", img_ol)
    cv2.imwrite(f"{OUT_SD}/{image_id}_SD_boxed.png", img_sd)

    print(f"✔ Processed {image_id}")



if __name__ == "__main__":
    # Loop over all JSON files
    for file in os.listdir(JSON_DIR):
        if file.endswith(".json"):
            image_id = file.replace(".json", "")
            draw_for_one(image_id)

    print("\n🎉 All images processed and saved in './data/boxed_images/'")
