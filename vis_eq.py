import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_contour_mask(img, method="canny", thresh1=50, thresh2=150):
    """
    Returns a binary mask of edges/contours for an input image.
    For RGB, works on grayscale.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if method == "canny":
        mask = cv2.Canny(gray, thresh1, thresh2)
    else:
        # Fallback to Sobel gradient
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        mask = (grad_mag > np.percentile(grad_mag, 95)).astype(np.uint8) * 255
    return mask


def compare_contour_masks(mask_rgb, mask_depth, tolerance=3):
    """
    Compare two binary contour masks, report coordinate-wise overlap and mismatch.
    Returns stats and a visualization mask.
    """
    # Dilate for tolerance matching (contours may be slightly misaligned)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    depth_dil = cv2.dilate(mask_depth, kernel)

    # Where RGB contour matches depth contour (allow small shift)
    match = (mask_rgb > 0) & (depth_dil > 0)
    rgb_only = (mask_rgb > 0) & ~match
    depth_only = (mask_depth > 0) & ~cv2.dilate(mask_rgb, kernel)

    n_match = np.sum(match)
    n_rgb = np.sum(mask_rgb > 0)
    n_depth = np.sum(mask_depth > 0)
    stats = {
        "n_match": int(n_match),
        "n_rgb_contour": int(n_rgb),
        "n_depth_contour": int(n_depth),
        "pct_rgb_match": 100.0 * n_match / n_rgb if n_rgb else 0.0,
        "pct_depth_match": 100.0 * n_match / n_depth if n_depth else 0.0,
    }

    vis = np.zeros((*mask_rgb.shape, 3), dtype=np.uint8)
    vis[rgb_only] = [255, 0, 0]
    vis[depth_only] = [0, 0, 255]
    vis[match] = [0, 255, 0]

    return stats, vis


def analyze_contour_alignment(
    rgb_path, depth_path, depth_range=(500, 1500), tolerance=3
):
    """
    Loads RGB/depth, extracts contours, compares them.
    """
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    if depth.ndim == 3:
        depth = depth.squeeze()
    # Optionally mask depth range
    depth_masked = np.zeros_like(depth)
    mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
    depth_masked[mask] = depth[mask]
    # Normalize for edge detection
    d_norm = cv2.normalize(depth_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Get contour masks
    mask_rgb = get_contour_mask(rgb)
    mask_depth = get_contour_mask(d_norm)

    stats, vis = compare_contour_masks(mask_rgb, mask_depth, tolerance=tolerance)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(mask_rgb, cmap="gray")
    plt.title("RGB contours")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mask_depth, cmap="gray")
    plt.title("Depth contours")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(vis)
    plt.title("Contour overlap: Green=match, Red=RGB, Blue=Depth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("contour_alignment.png", dpi=150)
    plt.close()
    print("Contour match stats:", stats)
    return stats


if __name__ == "__main__":
    analyze_contour_alignment(
        rgb_path="captures/000_rgb.png",
        depth_path="captures/000_depth.npy",
        depth_range=(500, 1500),
        tolerance=3,  # pixels allowed for shift
    )
