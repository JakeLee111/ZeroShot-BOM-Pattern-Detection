import cv2
import numpy as np

# =========================================================
# LOAD IMAGES
# =========================================================

pattern = cv2.imread("examples/pattern3.png", 0)   # grayscale
drawing = cv2.imread("examples/drawing.jpg", 0)   # grayscale

if pattern is None:
    raise ValueError("Pattern image not found at examples/pattern.jpg")

if drawing is None:
    raise ValueError("Drawing image not found at examples/drawing.jpg")

# =========================================================
# OUTPUT IMAGE (BGR for drawing colored boxes)
# =========================================================

output = cv2.cvtColor(drawing.copy(), cv2.COLOR_GRAY2BGR)

# =========================================================
# IMAGE DIMENSIONS
# =========================================================

H, W = drawing.shape
h0, w0 = pattern.shape

print("====================================")
print("IMAGE INFORMATION")
print("====================================")

print(f"Drawing : {W}x{H}")
print(f"Pattern : {w0}x{h0}")

# =========================================================
# ESTIMATE TARGET SCALE
# Assume the pattern appears at ~EXPECTED_OBJECT_WIDTH pixels
# wide in the drawing. Adjust this if results are off.
# =========================================================

EXPECTED_OBJECT_WIDTH = 60

estimated_scale = EXPECTED_OBJECT_WIDTH / w0

print("\nEstimated scale:", estimated_scale)

# =========================================================
# SEARCH RANGE: ±40% around estimated scale, 30 steps
# =========================================================

SCALES = np.linspace(
    estimated_scale * 0.4,
    estimated_scale * 1.2,
    30
)

# =========================================================
# ROTATION ANGLES (degrees)
# Tests 4 cardinal rotations to handle axis-aligned symbols
# =========================================================

ROTATIONS = [0, 90, 180, 270]

# Minimum similarity score to count as a match (0.0–1.0)
THRESHOLD = 0.8

boxes = []
scores = []

# =========================================================
# TEMPLATE MATCHING LOOP
# For each rotation × scale combination, run matchTemplate
# and collect all positions above THRESHOLD
# =========================================================

print("\n====================================")
print("RUNNING MATCHING")
print("====================================")

for angle in ROTATIONS:

    print("\n====================================")
    print(f"ROTATION: {angle} degrees")
    print("====================================")

    # =====================================================
    # ROTATE PATTERN
    # =====================================================

    if angle == 0:
        rotated_pattern = pattern.copy()

    elif angle == 90:
        rotated_pattern = cv2.rotate(pattern, cv2.ROTATE_90_CLOCKWISE)

    elif angle == 180:
        rotated_pattern = cv2.rotate(pattern, cv2.ROTATE_180)

    elif angle == 270:
        rotated_pattern = cv2.rotate(pattern, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # =====================================================
    # SCALE SEARCH
    # =====================================================

    for scale in SCALES:

        resized_pattern = cv2.resize(
            rotated_pattern,
            None,
            fx=scale,
            fy=scale
        )

        h, w = resized_pattern.shape

        # Skip if pattern is too small or larger than drawing
        if h < 10 or w < 10:
            continue
        if h > H or w > W:
            continue

        print(f"\nScale: {scale:.4f}")
        print(f"Pattern size: {w}x{h}")

        # =================================================
        # MATCH TEMPLATE
        # TM_CCOEFF_NORMED: normalized cross-correlation
        # Score range: [-1, 1], higher = more similar
        # =================================================

        result = cv2.matchTemplate(
            drawing,
            resized_pattern,
            cv2.TM_CCOEFF_NORMED
        )

        max_similarity = result.max()
        print(f"Max similarity: {max_similarity:.4f}")

        # =================================================
        # COLLECT MATCHES ABOVE THRESHOLD
        # =================================================

        locations = np.where(result >= THRESHOLD)
        match_count = 0

        for pt in zip(*locations[::-1]):
            x, y = pt
            score = result[y, x]
            boxes.append([x, y, w, h])
            scores.append(float(score))
            match_count += 1

        print(f"Matches: {match_count}")

# =========================================================
# NON-MAXIMUM SUPPRESSION
# Removes duplicate overlapping boxes, keeping highest score
# IoU threshold 0.3: boxes overlapping >30% area are merged
# =========================================================

print("\n====================================")
print("RUNNING NMS")
print("====================================")

print("Raw detections:", len(boxes))

indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=THRESHOLD,
    nms_threshold=0.3
)

# =========================================================
# DRAW RESULTS
# =========================================================

final_count = 0

if len(indices) > 0:

    for i in indices.flatten():

        x, y, w, h = boxes[i]
        score = scores[i]

        # Red bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Confidence score above box
        cv2.putText(
            output,
            f"{score:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        final_count += 1

# =========================================================
# SAVE OUTPUT
# =========================================================

cv2.imwrite("output.png", output)

# =========================================================
# FINAL REPORT
# =========================================================

print("\n====================================")
print("FINAL REPORT")
print("====================================")

print("Final detections:", final_count)
print("\nOutput saved as output.png")

print("\nPipeline:")
print("1. Rotation-aware Template Matching")
print("2. Multi-scale Search")
print("3. Non-Max Suppression")