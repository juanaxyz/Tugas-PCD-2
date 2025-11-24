from PIL import Image
import random


def add_white_speckles(input_path, output_path, amount=500):
    img = Image.open(input_path).convert("RGB")
    pixels = img.load()
    width, height = img.size

    for _ in range(amount):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixels[x, y] = (255, 255, 255)  # titik putih

    img.save(output_path)
    print("Selesai, tersimpan:", output_path)


# contoh pemakaian
add_white_speckles("images\input\input-cars.jpg", "output.jpg", amount=2000)
