from PIL import Image
import matplotlib.pyplot as plt
import math
from main import apply_kernel


def rgb_to_grayscale(img_path: str) -> Image:
    """Convert RGB image to grayscale using standard luminosity formula."""
    img = Image.open(img_path)
    px = img.load()
    canvas = Image.new("L", (img.width, img.height))
    px_new = canvas.load()

    for y in range(img.height):
        for x in range(img.width):

            r, g, b = px[x, y]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            px_new[x, y] = gray

    # canvas.show()
    return canvas


def load_image(img_path):
    """Load image and convert to grayscale if needed. Returns (grayscale, color) tuple."""
    img = Image.open(img_path)
    img_gray = img.copy()
    img_color = img.copy()

    # Use getpixel to check if image is RGB (tuple) or grayscale (int)
    if isinstance(img.getpixel((0, 0)), tuple):
        img_gray = rgb_to_grayscale(img_path)

    return img_gray, img_color


kernel_sobel = {
    "sobel_v": [[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]],
    "sobel_h": [
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ]
}


def sobel_edge_detection(img_gray):
    """Apply Sobel edge detection to grayscale image."""
    px = img_gray.load()
    w, h = img_gray.size

    img_gx = apply_kernel(
        px, w, h, kernel_sobel["sobel_h"], padding="replicate")
    img_gy = apply_kernel(
        px, w, h, kernel_sobel['sobel_v'], padding="replicate"
    )

    px_h = img_gx.load()
    px_v = img_gy.load()

    result = Image.new("L", (w, h))
    px_res = result.load()

    for x in range(w):
        for y in range(h):
            gx = px_h[x, y]
            gy = px_v[x, y]

            magnitude = math.sqrt(gx**2 + gy**2)
            px_res[x, y] = int(min(255, magnitude))

    return result


def threshold_image(img_gray, threshold=50):
    """Return binary PIL 'L' image with 0/255 values."""
    w, h = img_gray.size
    pixels = img_gray.load()
    out = Image.new("L", (w, h))
    out_px = out.load()
    for y in range(h):
        for x in range(w):
            out_px[x, y] = 255 if pixels[x, y] > threshold else 0
    return out


def connected_components(binary_img):
    """Label connected components (4-connectivity) using union-find.
    Input: PIL 'L' binary image (0/255). Returns (labels, label_count) where labels is
    a 2D list of ints (H x W) with labels 0..n, and label_count is number of labels.
    """
    w, h = binary_img.size
    px = binary_img.load()

    # mask ke true/false
    mask = [[px[x, y] != 0 for x in range(w)] for y in range(h)]

    labels = [[0 for _ in range(w)] for _ in range(h)]

    parent = [0]

    def make_set():
        parent.append(len(parent))
        return len(parent) - 1

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # First pass
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            neighbors = []
            # up
            if y > 0 and labels[y - 1][x] > 0:
                neighbors.append(labels[y - 1][x])
            # left
            if x > 0 and labels[y][x - 1] > 0:
                neighbors.append(labels[y][x - 1])

            if not neighbors:
                lab = make_set()
                labels[y][x] = lab
            else:
                lab = min(neighbors)
                labels[y][x] = lab
                for other in neighbors:
                    if other != lab:
                        union(other, lab)

    # Second pass
    root_map = {}
    next_label = 0

    for y in range(h):
        for x in range(w):
            lab = labels[y][x]
            if lab == 0:
                continue
            root = find(lab)
            if root not in root_map:
                next_label += 1
                root_map[root] = next_label
            labels[y][x] = root_map[root]

    return labels, next_label


def filter_by_area(labels, min_area=50, max_area=5000):
    """Filter components by area; labels is 2D list; return remapped 2D list of
    sequential labels and count.
    """
    h = len(labels)
    w = len(labels[0]) if h > 0 else 0
    counts = {}
    # count areas
    for y in range(h):
        for x in range(w):
            lab = labels[y][x]
            if lab == 0:
                continue
            # hitung jumlah komponen berdasarkan key label
            counts[lab] = counts.get(lab, 0) + 1

    # tentukan label yang valid berdasarkan area
    valid_labels = {lab for lab, cnt in counts.items() if min_area <=
                    cnt <= max_area}
    new_label_map = {}
    new_label = 0
    filtered = [[0 for _ in range(w)] for _ in range(h)]

    for lab in sorted(valid_labels):
        new_label += 1
        new_label_map[lab] = new_label

    for y in range(h):
        for x in range(w):
            lab = labels[y][x]
            if lab in new_label_map:
                filtered[y][x] = new_label_map[lab]

    return filtered, new_label


def labels_to_color_image(labels):
    """Convert a 2D label list to an RGB PIL image (colors for each label).
    Label 0 -> black; label n -> color from simple palette.
    """
    h = len(labels)
    w = len(labels[0]) if h > 0 else 0
    img = Image.new("RGB", (w, h))
    px = img.load()

    def color_for_label(l):
        if l == 0:
            return (0, 0, 0)
        r = (l * 97) % 256
        g = (l * 57) % 256
        b = (l * 37) % 256
        return (r, g, b)

    for y in range(h):
        for x in range(w):
            px[x, y] = color_for_label(labels[y][x])
    return img


def count_rice_grains(
    image_path,
    threshold=25,
    min_area=500,
    max_area=8000
):
    """Main pipeline to count rice grains without gaussian blur.
    Returns the count and saves result image.
    """
    print("Memuat gambar...")
    img_gray, img_color = load_image(image_path)

    print("Deteksi tepi (Sobel)...")
    edges = sobel_edge_detection(img_gray)

    print("Thresholding...")
    binary = threshold_image(edges, threshold)

    print("Labeling komponen (connected components)...")
    labels, num = connected_components(binary)
    print(f"Komponen ditemukan (total labels): {num}")

    print("Filter berdasarkan ukuran area...")
    filtered_labels, rice_count = filter_by_area(
        labels, min_area, max_area)
    print(f"Butir yang valid setelah filter: {rice_count}")

    # visualization
    color_labels_img = labels_to_color_image(filtered_labels)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_color)
    axes[0, 0].set_title('Gambar Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('Edge Detection (Sobel)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Image (Thresholding)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(color_labels_img)
    axes[1, 1].set_title(f'Connected Components\n(Jumlah: {rice_count} butir)')
    axes[1, 1].axis('off')

    # overlay original and bounding boxes
    overlay = img_color.copy()
    overlay_px = overlay.load()

    # find boxes per label
    boxes = {}
    h = len(filtered_labels)
    w = len(filtered_labels[0]) if h > 0 else 0
    for y in range(h):
        for x in range(w):
            lab = filtered_labels[y][x]
            if lab == 0:
                continue
            if lab not in boxes:
                boxes[lab] = [x, y, x, y]
            else:
                L = boxes[lab]
                if x < L[0]:
                    L[0] = x
                if y < L[1]:
                    L[1] = y
                if x > L[2]:
                    L[2] = x
                if y > L[3]:
                    L[3] = y

    # draw boxes (green) and centers (red)
    for lab, box in boxes.items():
        x1, y1, x2, y2 = box
        # draw horizontal edges
        for x in range(x1, x2 + 1):
            if 0 <= x < w and 0 <= y1 < h:
                overlay_px[x, y1] = (0, 255, 0)
            if 0 <= x < w and 0 <= y2 < h:
                overlay_px[x, y2] = (0, 255, 0)
        for y in range(y1, y2 + 1):
            if 0 <= x1 < w and 0 <= y < h:
                overlay_px[x1, y] = (0, 255, 0)
            if 0 <= x2 < w and 0 <= y < h:
                overlay_px[x2, y] = (0, 255, 0)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    overlay_px[nx, ny] = (255, 0, 0)

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Hasil {rice_count} butir')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('./images/output/hasil_deteksi_beras.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return rice_count


if __name__ == "__main__":
    image_path = "./images/input/beras_2.jpg"

    try:
        jumlah_beras = count_rice_grains(
            image_path,
            threshold=25,
            min_area=800,
            max_area=9000
        )

        print(f"\n{'='*50}")
        print(f"HASIL AKHIR: Terdeteksi {jumlah_beras} butir beras")
        print(f"{'='*50}")

    except FileNotFoundError:
        print(f"Error: File '{image_path}' tidak ditemukan!")
    except Exception as e:
        print(f"Error: {e}")
