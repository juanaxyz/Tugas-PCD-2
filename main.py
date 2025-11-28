from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

list_kernels = {
    # LOW-PASS FILTER (SMOOTHING)
    "mean": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    "gaussian": [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ],
    # HIGH-PASS FILTER (EDGE DETECTION)
    "laplacian_1": [
        [0, -1,  0],
        [-1,  4, -1],
        [0, -1,  0]
    ],
    "laplacian_2": [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ],
    "sobel_horizontal": [
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ],
    "sobel_vertical": [
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ],
    "outline": [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ],
    # HIGH-BOOST FILTER (SHARPENING)
    "sharpen": [
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ],
    # SPECIAL FILTER: EMBOSS
    "emboss": [
        [-2, -1, 0],
        [-1,  1, 1],
        [0,  1, 2]
    ],
    # SPECIAL FILTER: MOTION BLUR
    "motion_blur_horizontal": [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]
}


def check_neighbor(x, y):
    """4-ketetanggaan (atas, bawah, kiri, kanan)"""
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def get_pixel(px_img, w, h, nx, ny, padding="zero"):
    """
    Mengambil nilai pixel dengan berbagai jenis padding.
    px_img : PixelAccess object dari PIL
    nx, ny : posisi pixel yang diminta
    padding : "zero", "replicate", "reflect", "wrap"
    """

    # === 1. Zero Padding (default) ===
    if padding == "zero":
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            # Cek apakah RGB atau grayscale
            try:
                sample = px_img[0, 0]
                if isinstance(sample, tuple):
                    return (0, 0, 0)
                else:
                    return 0
            except:
                return 0
        return px_img[nx, ny]

    # === 2. Replicate Padding ===
    elif padding == "replicate":
        nx_clamped = max(0, min(nx, w - 1))
        ny_clamped = max(0, min(ny, h - 1))
        return px_img[nx_clamped, ny_clamped]

    # === 3. Reflect Padding (mirror) ===
    elif padding == "reflect":
        # untuk x
        if nx < 0:
            nx_reflect = -nx
        elif nx >= w:
            nx_reflect = 2*w - nx - 2
        else:
            nx_reflect = nx

        # untuk y
        if ny < 0:
            ny_reflect = -ny
        elif ny >= h:
            ny_reflect = 2*h - ny - 2
        else:
            ny_reflect = ny

        # Clamp ke batas valid
        nx_reflect = max(0, min(nx_reflect, w - 1))
        ny_reflect = max(0, min(ny_reflect, h - 1))

        return px_img[nx_reflect, ny_reflect]

    # === 4. Wrap Padding (cyclic) ===
    elif padding == "wrap":
        nx_wrap = nx % w
        ny_wrap = ny % h
        return px_img[nx_wrap, ny_wrap]

    # default
    else:
        return 0


def apply_kernel(px_img, w, h, kernel, padding="zero"):
    """
    Menerapkan kernel konvolusi pada gambar.
    px_img: PixelAccess object dari PIL
    """
    # Cek apakah gambar berwarna atau grayscale
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    k_size = len(kernel)
    offset = k_size // 2

    # Hitung sum kernel untuk normalisasi (jika diperlukan)
    k_sum = sum(sum(row) for row in kernel)

    if is_color:
        # Buat gambar output berwarna
        out_img = Image.new("RGB", (w, h))
        out_px = out_img.load()

        for y in range(h):
            for x in range(w):
                total_r = 0.0
                total_g = 0.0
                total_b = 0.0

                for i in range(-offset, offset + 1):
                    for j in range(-offset, offset + 1):
                        nx = x + j
                        ny = y + i
                        pv = get_pixel(px_img, w, h, nx, ny, padding)
                        kval = kernel[i + offset][j + offset]

                        if isinstance(pv, tuple):
                            total_r += kval * pv[0]
                            total_g += kval * pv[1]
                            total_b += kval * pv[2]
                        else:
                            total_r += kval * pv
                            total_g += kval * pv
                            total_b += kval * pv

                # Normalisasi jika kernel sum > 0 (untuk mean/gaussian)
                if k_sum > 0:
                    total_r /= k_sum
                    total_g /= k_sum
                    total_b /= k_sum

                # Clamp ke range 0-255
                r = max(0, min(255, int(total_r)))
                g = max(0, min(255, int(total_g)))
                b = max(0, min(255, int(total_b)))

                out_px[x, y] = (r, g, b)

    else:
        # Buat gambar output grayscale
        out_img = Image.new("L", (w, h))
        out_px = out_img.load()

        for y in range(h):
            for x in range(w):
                total = 0.0

                for i in range(-offset, offset + 1):
                    for j in range(-offset, offset + 1):
                        nx = x + j
                        ny = y + i
                        pv = get_pixel(px_img, w, h, nx, ny, padding)
                        kval = kernel[i + offset][j + offset]
                        total += kval * pv

                # Normalisasi jika kernel sum > 0
                if k_sum > 0:
                    total /= k_sum

                # Clamp ke range 0-255
                out_px[x, y] = max(0, min(255, int(total)))

    return out_img


def filter_batas(px_img, w, h, padding="zero"):
    """Filter batas: clamp pixel ke min/max tetangga"""
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    if is_color:
        out_img = Image.new("RGB", (w, h))
    else:
        out_img = Image.new("L", (w, h))

    out_px = out_img.load()

    for y in range(h):
        for x in range(w):
            neighbors = check_neighbor(x, y)

            if is_color:
                r_val = []
                g_val = []
                b_val = []

                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        r, g, b = px_img[nx, ny]
                        r_val.append(r)
                        g_val.append(g)
                        b_val.append(b)

                if not r_val:
                    out_px[x, y] = px_img[x, y]
                    continue

                min_r = min(r_val)
                min_g = min(g_val)
                min_b = min(b_val)
                max_r = max(r_val)
                max_g = max(g_val)
                max_b = max(b_val)

                r, g, b = px_img[x, y]
                new_r = max(min_r, min(r, max_r))
                new_g = max(min_g, min(g, max_g))
                new_b = max(min_b, min(b, max_b))

                out_px[x, y] = (new_r, new_g, new_b)
            else:
                vals = []
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        vals.append(px_img[nx, ny])

                if not vals:
                    out_px[x, y] = px_img[x, y]
                    continue

                min_v = min(vals)
                max_v = max(vals)
                v = px_img[x, y]
                new_v = max(min_v, min(v, max_v))

                out_px[x, y] = new_v

    return out_img


def filter_batas_min(px_img, w, h, padding="zero"):
    """Filter batas min"""
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    if is_color:
        out_img = Image.new("RGB", (w, h))
    else:
        out_img = Image.new("L", (w, h))

    out_px = out_img.load()

    for y in range(h):
        for x in range(w):
            neighbors = check_neighbor(x, y)

            if is_color:
                r_val = []
                g_val = []
                b_val = []

                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        r, g, b = px_img[nx, ny]
                        r_val.append(r)
                        g_val.append(g)
                        b_val.append(b)

                if not r_val:
                    out_px[x, y] = px_img[x, y]
                    continue

                r, g, b = px_img[x, y]
                new_r = min(min(r_val), r)
                new_g = min(min(g_val), g)
                new_b = min(min(b_val), b)

                out_px[x, y] = (new_r, new_g, new_b)
            else:
                vals = []
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        vals.append(px_img[nx, ny])

                if not vals:
                    out_px[x, y] = px_img[x, y]
                    continue

                v = px_img[x, y]
                new_v = min(min(vals), v)
                out_px[x, y] = new_v

    return out_img


def filter_batas_max(px_img, w, h, padding="zero"):
    """Filter batas max"""
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    if is_color:
        out_img = Image.new("RGB", (w, h))
    else:
        out_img = Image.new("L", (w, h))

    out_px = out_img.load()

    for y in range(h):
        for x in range(w):
            neighbors = check_neighbor(x, y)

            if is_color:
                r_val = []
                g_val = []
                b_val = []

                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        r, g, b = px_img[nx, ny]
                        r_val.append(r)
                        g_val.append(g)
                        b_val.append(b)

                if not r_val:
                    out_px[x, y] = px_img[x, y]
                    continue

                r, g, b = px_img[x, y]
                new_r = max(max(r_val), r)
                new_g = max(max(g_val), g)
                new_b = max(max(b_val), b)

                out_px[x, y] = (new_r, new_g, new_b)
            else:
                vals = []
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        vals.append(px_img[nx, ny])

                if not vals:
                    out_px[x, y] = px_img[x, y]
                    continue

                v = px_img[x, y]
                new_v = max(max(vals), v)
                out_px[x, y] = new_v

    return out_img


def filter_mean(px_img, w, h, padding="zero"):
    """Filter mean menggunakan 4-tetangga"""
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    if is_color:
        out_img = Image.new("RGB", (w, h))
    else:
        out_img = Image.new("L", (w, h))

    out_px = out_img.load()

    for y in range(h):
        for x in range(w):
            neighbors = check_neighbor(x, y)

            if is_color:
                r_val = []
                g_val = []
                b_val = []

                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        r, g, b = px_img[nx, ny]
                        r_val.append(r)
                        g_val.append(g)
                        b_val.append(b)

                if not r_val:
                    out_px[x, y] = px_img[x, y]
                    continue

                mean_r = sum(r_val) // len(r_val)
                mean_g = sum(g_val) // len(g_val)
                mean_b = sum(b_val) // len(b_val)

                out_px[x, y] = (mean_r, mean_g, mean_b)
            else:
                vals = []
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        vals.append(px_img[nx, ny])

                if not vals:
                    out_px[x, y] = px_img[x, y]
                    continue

                mean_v = sum(vals) // len(vals)
                out_px[x, y] = mean_v

    return out_img


def filter_median(px_img, w, h, padding="zero"):
    """Filter median menggunakan 4-tetangga"""
    sample = px_img[0, 0]
    is_color = isinstance(sample, tuple)

    if is_color:
        out_img = Image.new("RGB", (w, h))
    else:
        out_img = Image.new("L", (w, h))

    out_px = out_img.load()

    for y in range(h):
        for x in range(w):
            neighbors = check_neighbor(x, y)

            if is_color:
                r_val = []
                g_val = []
                b_val = []

                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        r, g, b = px_img[nx, ny]
                        r_val.append(r)
                        g_val.append(g)
                        b_val.append(b)

                if not r_val:
                    out_px[x, y] = px_img[x, y]
                    continue

                median_r = sorted(r_val)[len(r_val) // 2]
                median_g = sorted(g_val)[len(g_val) // 2]
                median_b = sorted(b_val)[len(b_val) // 2]

                out_px[x, y] = (median_r, median_g, median_b)
            else:
                vals = []
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        vals.append(px_img[nx, ny])

                if not vals:
                    out_px[x, y] = px_img[x, y]
                    continue

                median_v = sorted(vals)[len(vals) // 2]
                out_px[x, y] = median_v

    return out_img


def show_images_matplotlib(input_img, result_img, filter_name, titles=("Input", "Result")):
    """Menampilkan dua gambar berdampingan menggunakan matplotlib"""
    if isinstance(input_img, str):
        input_img = Image.open(input_img)
    if isinstance(result_img, str):
        result_img = Image.open(result_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_img)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(result_img)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    fig.canvas.manager.set_window_title(filter_name)
    plt.tight_layout()
    plt.show()


def print_menu():
    """Menampilkan menu pilihan filter"""
    print("\n" + "="*60)
    print("IMAGE FILTERING PROGRAM".center(60))
    print("="*60)
    print("\n[KERNEL-BASED FILTERS]")
    print("1.  Mean (Low-pass)")
    print("2.  Gaussian (Low-pass)")
    print("3.  Laplacian 1 (High-pass)")
    print("4.  Laplacian 2 (High-pass)")
    print("5.  Sobel Horizontal (Edge Detection)")
    print("6.  Sobel Vertical (Edge Detection)")
    print("7.  Outline (Edge Detection)")
    print("8.  Sharpen (High-boost)")
    print("9.  Emboss (Special Effect)")
    print("10. Motion Blur Horizontal")

    print("\n[NEIGHBOR-BASED FILTERS]")
    print("11. Filter Mean (4-neighbors)")
    print("12. Filter Median (4-neighbors)")
    print("13. Filter Batas (Boundary)")
    print("14. Filter Batas Min")
    print("15. Filter Batas Max")

    print("\n0. Keluar")
    print("="*60)


def choose_padding():
    """Memilih jenis padding"""
    print("\n[PILIH PADDING]")
    print("1. Zero Padding (default)")
    print("2. Replicate Padding")
    print("3. Reflect Padding")
    print("4. Wrap Padding")

    while True:
        choice = input("Pilih padding (1-4) [default: 1]: ").strip()
        if choice == "" or choice == "1":
            return "zero"
        elif choice == "2":
            return "replicate"
        elif choice == "3":
            return "reflect"
        elif choice == "4":
            return "wrap"
        else:
            print("Pilihan tidak valid! Silakan pilih 1-4.")


def process_filter(img, px, w, h, filter_choice, padding):
    """Memproses gambar dengan filter yang dipilih"""

    kernel_filters = {
        1: ("mean", "Kernel - Mean"),
        2: ("gaussian", "Kernel - Gaussian"),
        3: ("laplacian_1", "Kernel - Laplacian 1"),
        4: ("laplacian_2", "Kernel - Laplacian 2"),
        5: ("sobel_horizontal", "Kernel - Sobel Horizontal"),
        6: ("sobel_vertical", "Kernel - Sobel Vertical"),
        7: ("outline", "Kernel - Outline"),
        8: ("sharpen", "Kernel - Sharpen"),
        9: ("emboss", "Kernel - Emboss"),
        10: ("motion_blur_horizontal", "Kernel - Motion Blur")
    }

    if filter_choice in kernel_filters:
        kernel_name, display_name = kernel_filters[filter_choice]
        print(f"\nMemproses dengan {display_name} (padding: {padding})...")
        result = apply_kernel(
            px, w, h, list_kernels[kernel_name], padding=padding)
        show_images_matplotlib(img, result, display_name)

    elif filter_choice == 11:
        print(f"\nMemproses dengan Filter Mean...")
        result = filter_mean(px, w, h, padding=padding)
        show_images_matplotlib(img, result, "Filter - Mean (4-neighbors)")

    elif filter_choice == 12:
        print(f"\nMemproses dengan Filter Median...")
        result = filter_median(px, w, h, padding=padding)
        show_images_matplotlib(img, result, "Filter - Median")

    elif filter_choice == 13:
        print(f"\nMemproses dengan Filter Batas...")
        result = filter_batas(px, w, h, padding=padding)
        show_images_matplotlib(img, result, "Filter - Batas")

    elif filter_choice == 14:
        print(f"\nMemproses dengan Filter Batas Min...")
        result = filter_batas_min(px, w, h, padding=padding)
        show_images_matplotlib(img, result, "Filter - Batas Min")

    elif filter_choice == 15:
        print(f"\nMemproses dengan Filter Batas Max...")
        result = filter_batas_max(px, w, h, padding=padding)
        show_images_matplotlib(img, result, "Filter - Batas Max")

    else:
        print("Pilihan filter tidak valid!")


if __name__ == "__main__":
    # Input path gambar
    print("\n" + "="*60)
    print("IMAGE FILTERING PROGRAM".center(60))
    print("="*60)

    img_path = input(
        "\nMasukkan path gambar [default: ./images/input/noise.jpg]: ").strip()
    if not img_path:
        img_path = "./images/input/noise.jpg"

    try:
        img = Image.open(img_path)
        px = img.load()
        w, h = img.size

        print(f"\n✓ Gambar berhasil dimuat!")
        print(f"  Ukuran: {w} x {h} pixels")
        print(f"  Mode: {img.mode}")

        while True:
            print_menu()

            choice = input("\nPilih filter (0-15): ").strip()

            if choice == "0":
                print("\nTerima kasih! Program selesai.")
                break

            try:
                filter_choice = int(choice)

                if filter_choice < 0 or filter_choice > 15:
                    print("Pilihan tidak valid! Silakan pilih 0-15.")
                    continue

                # Pilih padding
                padding = choose_padding()

                # Proses filter
                process_filter(img, px, w, h, filter_choice, padding)

                # Tanya apakah ingin mencoba filter lain
                again = input(
                    "\nCoba filter lain? (y/n) [y]: ").strip().lower()
                if again == 'n':
                    print("\nTerima kasih! Program selesai.")
                    break

            except ValueError:
                print("Input tidak valid! Masukkan angka 0-15.")
            except Exception as e:
                print(f"Error saat memproses: {e}")

    except FileNotFoundError:
        print(f"\n✗ Error: File tidak ditemukan di path: {img_path}")
        print("  Pastikan path file benar dan file ada.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
