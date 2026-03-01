import cv2
import numpy as np

def calculate_density(image_path, person_coords, grid_size=8):
    """
    TRUE density = people per grid cell (area-normalized in the sense of per-cell count).
    Returns a (grid_size x grid_size) float32 array.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("❌ density.py: Could not read image:", image_path)
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    h, w = image.shape[:2]
    density = np.zeros((grid_size, grid_size), dtype=np.float32)

    if person_coords is None or len(person_coords) == 0:
        return density

    cell_w = w / float(grid_size)
    cell_h = h / float(grid_size)

    # Safe coords + counting
    for (cx, cy) in person_coords:
        # clip coords to image bounds first
        cx = int(min(max(cx, 0), w - 1))
        cy = int(min(max(cy, 0), h - 1))

        col = int(cx / cell_w)
        row = int(cy / cell_h)

        # clip to grid bounds
        col = min(max(col, 0), grid_size - 1)
        row = min(max(row, 0), grid_size - 1)

        density[row, col] += 1.0

    return density


def generate_heatmap(image_path, density_map, output_path):
    """
    Creates a JET heatmap overlay and saves to output_path.
    This is broadcast-safe (no numpy alpha shape issues).
    """
    image = cv2.imread(image_path)
    if image is None:
        print("❌ density.py: Could not read image for heatmap:", image_path)
        return

    h, w = image.shape[:2]

    dm = density_map.astype(np.float32)
    if dm.size == 0:
        dm = np.zeros((8, 8), dtype=np.float32)

    # Normalize 0..255 for colormap
    mx = float(np.max(dm)) if np.max(dm) > 0 else 1.0
    norm = (dm / mx * 255.0).astype(np.uint8)

    # Resize grid -> full image size
    heat = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(image, 0.60, heat_color, 0.40, 0)

    # Optional: draw grid lines
    grid_size = dm.shape[0]
    for i in range(1, grid_size):
        x = int(i * w / grid_size)
        y = int(i * h / grid_size)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)

    # Optional: label cell counts (small text)
    cell_w = w / float(grid_size)
    cell_h = h / float(grid_size)
    for r in range(grid_size):
        for c in range(grid_size):
            val = int(round(float(dm[r, c])))
            if val <= 0:
                continue
            tx = int(c * cell_w + 6)
            ty = int(r * cell_h + 18)
            cv2.putText(overlay, str(val), (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, str(val), (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)


def assess_risk(person_count, density_map):
    """
    Simple risk rules using both total count and max cell density.
    Returns: (risk_text, recommendations_list)
    """
    max_cell = float(np.max(density_map)) if density_map is not None and density_map.size else 0.0

    # You can tune these thresholds for your demo
    if person_count <= 10 and max_cell <= 3:
        return "Safe 🟢", [
            "Maintain normal monitoring",
            "Ensure entry/exit routes are clear"
        ]
    elif person_count <= 35 and max_cell <= 6:
        return "Crowded 🟡", [
            "Deploy additional staff near hotspots",
            "Control entry rate if crowd grows",
            "Keep emergency paths open"
        ]
    else:
        return "Dangerous 🔴", [
            "Restrict new entry immediately",
            "Disperse crowd from hotspot areas",
            "Prepare evacuation protocol"
        ]
