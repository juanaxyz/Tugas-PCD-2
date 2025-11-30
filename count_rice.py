import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    """Memuat gambar dan konversi ke grayscale"""
    img = Image.open(image_path)
    img_gray = img.convert('L')
    return np.array(img_gray), np.array(img)


def apply_gaussian_blur(img, kernel_size=5, sigma=1.0):
    """Gaussian blur manual untuk mengurangi noise"""
    height, width = img.shape
    pad = kernel_size // 2
    blurred = np.zeros_like(img, dtype=float)

    # Buat Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - pad
            y = j - pad
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Apply convolution
    img_padded = np.pad(img, pad, mode='edge')
    for i in range(height):
        for j in range(width):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            blurred[i, j] = np.sum(region * kernel)

    return blurred.astype(np.uint8)


def sobel_edge_detection(img):
    """Deteksi tepi menggunakan operator Sobel manual"""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    height, width = img.shape
    gradient_x = np.zeros_like(img, dtype=float)
    gradient_y = np.zeros_like(img, dtype=float)

    # Padding
    img_padded = np.pad(img, 1, mode='edge')

    # Konvolusi manual
    for i in range(height):
        for j in range(width):
            region = img_padded[i:i+3, j:j+3]
            gradient_x[i, j] = np.sum(region * sobel_x)
            gradient_y[i, j] = np.sum(region * sobel_y)

    # Hitung magnitude
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    return magnitude


def threshold_image(img, threshold=50):
    """Thresholding manual untuk binarisasi"""
    binary = np.zeros_like(img)
    binary[img > threshold] = 255
    return binary


def connected_components(binary_img):
    """Label connected components using an iterative, union-find two-pass algorithm.
    This implementation is significantly faster and avoids costly python-level
    recursive equivalence resolution used previously.
    Returns: (labels, num_components)
    """
    height, width = binary_img.shape
    # Convert to boolean mask (True==object) for convenience
    if binary_img.dtype == np.uint8:
        mask = binary_img > 0
    else:
        mask = binary_img.astype(bool)

    labels = np.zeros((height, width), dtype=np.int32)

    # Union-Find (Disjoint Set) implementation
    parent = [0]  # index 0 unused; parent[i] = parent label of i

    def make_set():
        label = len(parent)
        parent.append(label)
        return label

    def find(x):
        # path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # attach larger root to smaller root to keep roots small
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # First pass: assign labels and record unions
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            neighbors = []
            # 4-connectivity: up, left
            if i > 0 and labels[i - 1, j] > 0:
                neighbors.append(labels[i - 1, j])
            if j > 0 and labels[i, j - 1] > 0:
                neighbors.append(labels[i, j - 1])

            if not neighbors:
                new_label = make_set()
                labels[i, j] = new_label
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                # union other neighbor labels with min label
                for other in neighbors:
                    if other != min_label:
                        union(other, min_label)

    # Second pass: flatten and relabel sequentially
    # Build mapping from root->new label
    max_label = len(parent) - 1
    root_map = np.zeros(max_label + 1, dtype=np.int32)
    next_label = 0

    for lab in range(1, max_label + 1):
        if parent[lab] == 0:
            continue
        root = find(lab)
        if root_map[root] == 0:
            next_label += 1
            root_map[root] = next_label
    # Now map labels array
    # flatten-labels and remap using find + root_map
    for i in range(height):
        for j in range(width):
            lab = labels[i, j]
            if lab > 0:
                root = find(lab)
                labels[i, j] = root_map[root]

    return labels, next_label


def filter_by_area(labels, min_area=50, max_area=5000):
    """Filter komponen berdasarkan area (ukuran)"""
    # Use numpy bincount for fast area counting
    flat = labels.flatten()
    max_label = flat.max()
    if max_label == 0:
        return labels, 0

    counts = np.bincount(flat)
    filtered_labels = np.zeros_like(labels)
    # Map old label -> new sequential label
    label_map = np.zeros(max_label + 1, dtype=np.int32)
    new_label = 0
    for lab in range(1, max_label + 1):
        area = counts[lab] if lab < len(counts) else 0
        if min_area <= area <= max_area:
            new_label += 1
            label_map[lab] = new_label

    if new_label == 0:
        return filtered_labels, 0

    # remap labels
    remapped = label_map[labels]
    return remapped, new_label

# Main process


def count_rice_grains(image_path):
    """Fungsi utama untuk menghitung butir beras"""
    print("Memuat gambar...")
    img_gray, img_color = load_image(image_path)

    print("Menerapkan Gaussian blur...")
    img_blurred = apply_gaussian_blur(img_gray, kernel_size=3, sigma=1.0)

    print("Mendeteksi tepi dengan Sobel...")
    edges = sobel_edge_detection(img_blurred)

    print("Thresholding...")
    binary = threshold_image(edges, threshold=25)

    print("Mencari connected components...")
    # Convert binary to 0/1 mask for the union-find labeling
    labels, num_components = connected_components(binary)

    print("Filtering berdasarkan ukuran...")
    filtered_labels, rice_count = filter_by_area(
        labels, min_area=350, max_area=8000)

    # Visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_color)
    axes[0, 0].set_title('Gambar Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_blurred, cmap='gray')
    axes[0, 1].set_title('Setelah Gaussian Blur')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('Edge Detection (Sobel)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Image (Thresholding)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(filtered_labels, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Connected Components\n(Jumlah: {rice_count} butir)')
    axes[1, 1].axis('off')

    # Overlay hasil pada gambar original
    overlay = img_color.copy()
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Hasil Deteksi: {rice_count} butir beras')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('./images/output/hasil_deteksi_beras.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return rice_count


# Jalankan program
if __name__ == "__main__":
    # Ganti dengan path gambar Anda
    image_path = "./images/input/beras.jpg"  # atau "beras.png"

    try:
        jumlah_beras = count_rice_grains(image_path)
        print(f"\n{'='*50}")
        print(f"HASIL AKHIR: Terdeteksi {jumlah_beras} butir beras")
        print(f"{'='*50}")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' tidak ditemukan!")
        print("Pastikan gambar berada di direktori yang sama dengan script ini.")
    except Exception as e:
        print(f"Error: {e}")
