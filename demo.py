import cv2
import gradio as gr
import numpy as np
import json

def detect(pattern_image, drawing_image):
    
    # =========================================================
    # LOAD IMAGES
    # =========================================================

    pattern = cv2.cvtColor(pattern_image, cv2.COLOR_RGB2GRAY)
    drawing = cv2.cvtColor(drawing_image, cv2.COLOR_RGB2GRAY)

    if pattern is None:
        raise ValueError("Pattern image not found")

    if drawing is None:
        raise ValueError("Drawing image not found")

    # =========================================================
    # OUTPUT IMAGE
    # =========================================================

    output = cv2.cvtColor(
        drawing.copy(),
        cv2.COLOR_GRAY2BGR
    )

    # =========================================================
    # IMAGE INFO
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
    # =========================================================

    EXPECTED_OBJECT_WIDTH = 60

    estimated_scale = EXPECTED_OBJECT_WIDTH / w0

    print("\nEstimated scale:", estimated_scale)

    # =========================================================
    # SEARCH AROUND ESTIMATED SCALE
    # =========================================================

    SCALES = np.linspace(
        estimated_scale * 0.4,
        estimated_scale * 1.2,
        30
    )

    # =========================================================
    # ROTATION ANGLES
    # =========================================================

    ROTATIONS = [
        0,
        90,
        180,
        270
    ]

    THRESHOLD = 0.8

    boxes = []
    scores = []

    # =========================================================
    # TEMPLATE MATCHING
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
            rotated_pattern = cv2.rotate(
                pattern,
                cv2.ROTATE_90_CLOCKWISE
            )

        elif angle == 180:
            rotated_pattern = cv2.rotate(
                pattern,
                cv2.ROTATE_180
            )

        elif angle == 270:
            rotated_pattern = cv2.rotate(
                pattern,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

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

            # Skip invalid size
            if h < 10 or w < 10:
                continue

            if h > H or w > W:
                continue

            print(f"\nScale: {scale:.4f}")
            print(f"Pattern size: {w}x{h}")

            # =================================================
            # MATCH TEMPLATE
            # =================================================

            result = cv2.matchTemplate(
                drawing,
                resized_pattern,
                cv2.TM_CCOEFF_NORMED
            )

            max_similarity = result.max()

            print(f"Max similarity: {max_similarity:.4f}")

            # =================================================
            # FIND MATCHES
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
    # NMS
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
    detections = []

    if len(indices) > 0:

        for i in indices.flatten():

            x, y, w, h = boxes[i]
            score = scores[i]

            # RED BOX
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            # SCORE
            cv2.putText(
                output,
                f"{score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": round(float(score), 4)
            })

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

    # =========================================================
    # JSON OUTPUT
    # =========================================================

    json_output = json.dumps(
        detections,
        indent=2
    )

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb, json_output

# GRADIO UI

demo = gr.Interface(
    fn=detect,

    inputs=[
        gr.Image(label="Pattern Image"),
        gr.Image(label="Drawing Image")
    ],

    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Bounding Boxes JSON")
    ],

    title="Pattern Detection Demo",

    description="""
Upload:
- Pattern image
- Drawing image

The system will:
- detect matching patterns
- draw bounding boxes
- return confidence scores
"""
)

# LAUNCH

demo.launch()