from PIL import Image
import os

folders = ["dataset/cats", "dataset/dogs"]

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = Image.open(path)
        rgb_img = img.convert("RGB")

        new_path = os.path.splitext(path)[0] + ".jpg"
        rgb_img.save(new_path, "JPEG")

        print("Converted:", new_path)