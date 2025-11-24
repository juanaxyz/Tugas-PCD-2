from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def check_neighbor(x, y):
    """4-ketetanggaan (atas, bawah, kiri, kanan)"""
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    # return [
    #     (x + 1, y),
    #     (x - 1, y),
    #     (x, y + 1),
    #     (x, y - 1),
    #     (x + 1, y + 1),
    #     (x - 1, y + 1),
    #     (x + 1, y - 1),
    #     (x - 1, y - 1),
    # ]


def filter_batas(img_path):
    img = Image.open(img_path)
    px = img.load()

    canvas = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    canvas_px = canvas.load()

    for x in range(img.width):
        for y in range(img.height):
            neighbors = check_neighbor(x, y)
            r_val = []
            g_val = []
            b_val = []

            for nx, ny in neighbors:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    r, g, b = px[nx, ny]
                    r_val.append(r)
                    g_val.append(g)
                    b_val.append(b)

            # Handle empty neighbor lists (piksel di tepi)
            if not r_val:
                canvas_px[x, y] = px[x, y]  # Gunakan nilai asli
                continue

            minR = min(r_val)
            minG = min(g_val)
            minB = min(b_val)
            maxR = max(r_val)
            maxG = max(g_val)
            maxB = max(b_val)

            # Process current pixel
            r, g, b = px[x, y]

            # Clamp values between min and max of neighbors
            new_r = minR if r < minR else maxR if r > maxR else r
            new_g = minG if g < minG else maxG if g > maxG else g
            new_b = minB if b < minB else maxB if b > maxB else b

            canvas_px[x, y] = (new_r, new_g, new_b)

    return canvas


def filter_batas_min(img_path):
    img = Image.open(img_path)
    px = img.load()

    canvas = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    canvas_px = canvas.load()

    for x in range(img.width):
        for y in range(img.height):
            neighbors = check_neighbor(x, y)
            r_val = []
            g_val = []
            b_val = []

            for nx, ny in neighbors:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    r, g, b = px[nx, ny]
                    r_val.append(r)
                    g_val.append(g)
                    b_val.append(b)

            # Handle empty neighbor lists (piksel di tepi)
            if not r_val:
                canvas_px[x, y] = px[x, y]  # Gunakan nilai asli
                continue

            minR = min(r_val)
            minG = min(g_val)
            minB = min(b_val)

            # Process current pixel
            r, g, b = px[x, y]

            # Clamp values to minimum of neighbors
            new_r = minR if r < minR else r
            new_g = minG if g < minG else g
            new_b = minB if b < minB else b

            canvas_px[x, y] = (new_r, new_g, new_b)

    return canvas


def filter_batas_max(img_path):
    img = Image.open(img_path)
    px = img.load()

    canvas = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    canvas_px = canvas.load()

    for x in range(img.width):
        for y in range(img.height):
            neighbors = check_neighbor(x, y)
            r_val = []
            g_val = []
            b_val = []

            for nx, ny in neighbors:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    r, g, b = px[nx, ny]
                    r_val.append(r)
                    g_val.append(g)
                    b_val.append(b)

            # Handle empty neighbor lists (piksel di tepi)
            if not r_val:
                canvas_px[x, y] = px[x, y]  # Gunakan nilai asli
                continue

            maxR = max(r_val)
            maxG = max(g_val)
            maxB = max(b_val)

            # Process current pixel
            r, g, b = px[x, y]

            # Clamp values to maximum of neighbors
            new_r = maxR if r > maxR else r
            new_g = maxG if g > maxG else g
            new_b = maxB if b > maxB else b

            canvas_px[x, y] = (new_r, new_g, new_b)

    return canvas


def filter_mean(img_path):
    img = Image.open(img_path)
    px = img.load()

    canvas = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    canvas_px = canvas.load()

    for x in range(img.width):
        for y in range(img.height):
            neighbors = check_neighbor(x, y)
            r_val = []
            g_val = []
            b_val = []

            for nx, ny in neighbors:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    r, g, b = px[nx, ny]
                    r_val.append(r)
                    g_val.append(g)
                    b_val.append(b)

            # Handle empty neighbor lists (piksel di tepi)
            if not r_val:
                canvas_px[x, y] = px[x, y]  # Gunakan nilai asli
                continue

            meanR = sum(r_val) // len(r_val)
            meanG = sum(g_val) // len(g_val)
            meanB = sum(b_val) // len(b_val)

            # Process current pixel
            r, g, b = px[x, y]

            # Clamp values to mean of neighbors
            new_r = meanR
            new_g = meanG
            new_b = meanB

            canvas_px[x, y] = (new_r, new_g, new_b)

    return canvas


def filter_median(img_path):
    img = Image.open(img_path)
    px = img.load()

    canvas = Image.new("RGB", (img.width, img.height), (0, 0, 0))
    canvas_px = canvas.load()

    for x in range(img.width):
        for y in range(img.height):
            neighbors = check_neighbor(x, y)
            r_val = []
            g_val = []
            b_val = []

            for nx, ny in neighbors:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    r, g, b = px[nx, ny]
                    r_val.append(r)
                    g_val.append(g)
                    b_val.append(b)

            # Handle empty neighbor lists (piksel di tepi)
            if not r_val:
                canvas_px[x, y] = px[x, y]  # Gunakan nilai asli
                continue

            medianR = sorted(r_val)[len(r_val) // 2]
            medianG = sorted(g_val)[len(g_val) // 2]
            medianB = sorted(b_val)[len(b_val) // 2]

            # Process current pixel
            r, g, b = px[x, y]

            # Clamp values to median of neighbors
            new_r = medianR
            new_g = medianG
            new_b = medianB

            canvas_px[x, y] = (new_r, new_g, new_b)

    return canvas


def show_images_matplotlib(
    input_img, result_img, filter_name, titles=("Input", "Result")
):
    """Display two images side-by-side using matplotlib.
    Accepts either PIL Image objects or file paths.
    """
    if isinstance(input_img, str):
        input_img = Image.open(input_img)
    if isinstance(result_img, str):
        result_img = Image.open(result_img)

    arr1 = np.array(input_img)
    arr2 = np.array(result_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(arr1)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(arr2)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    fig.canvas.manager.set_window_title(filter_name)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    noise = "./images/input/output.jpg"
    result = filter_batas(noise)
    show_images_matplotlib(noise, result, "Filter Batas")
    # filter_batas_min(test_path)
    # filter_batas_max(test_path)
    result = filter_mean(noise)
    show_images_matplotlib(noise, result, "Filter Mean")

    result = filter_median(noise)
    show_images_matplotlib(noise, result, "Filter Median")
