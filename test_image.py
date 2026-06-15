from PIL import Image

img = Image.open("dataset/cats/cat1.jpg")

print("Format:", img.format)
print("Size:", img.size)