import cv2
import gradio as gr
import numpy as np
import json
import os

# =========================================================
# CORE DETECTION FUNCTION
# =========================================================

def detect(pattern_image, drawing_image, expected_width=60, threshold=0.8):
    """
    Detect pattern occurrences in a drawing using multi-scale,
    rotation-aware template matching with NMS post-processing.

    Args:
        pattern_image (np.ndarray): RGB pattern image (from Gradio)
        drawing_image (np.ndarray): RGB drawing image (from Gradio)
        expected_width (int): Expected width (px) of pattern in drawing
        threshold (float): Minimum match confidence (0.0–1.0)

    Returns:
        output_rgb (np.ndarray): Annotated drawing with bounding boxes
        json_output (str): JSON string with detection details
        summary (str): Human-readable detection summary
    """

    # =========================================================
    # VALIDATE INPUTS
    # =========================================================

    if pattern_image is None:
        raise gr.Error("Please upload a pattern image.")
    if drawing_image is None:
        raise gr.Error("Please upload a drawing image.")

    # =========================================================
    # CONVERT TO GRAYSCALE
    # =========================================================

    pattern = cv2.cvtColor(pattern_image, cv2.COLOR_RGB2GRAY)
    drawing = cv2.cvtColor(drawing_image, cv2.COLOR_RGB2GRAY)

    # =========================================================
    # OUTPUT IMAGE (BGR for OpenCV drawing)
    # =========================================================

    output = cv2.cvtColor(drawing.copy(), cv2.COLOR_GRAY2BGR)

    # =========================================================
    # IMAGE DIMENSIONS
    # =========================================================

    H, W = drawing.shape
    h0, w0 = pattern.shape

    # =========================================================
    # ESTIMATE SCALE FROM EXPECTED WIDTH
    # =========================================================

    estimated_scale = expected_width / w0

    # =========================================================
    # DEFINE SCALE SEARCH RANGE (±40% around estimate)
    # =========================================================

    SCALES = np.linspace(
        estimated_scale * 0.4,
        estimated_scale * 1.2,
        30
    )

    # =========================================================
    # ROTATION ANGLES TO TEST
    # =========================================================

    ROTATIONS = [0, 90, 180, 270]

    ROTATE_CODES = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    boxes = []
    scores = []

    # =========================================================
    # MAIN MATCHING LOOP: ROTATION × SCALE
    # =========================================================

    for angle in ROTATIONS:

        # Rotate pattern
        if angle == 0:
            rotated_pattern = pattern.copy()
        else:
            rotated_pattern = cv2.rotate(pattern, ROTATE_CODES[angle])

        # Scale search
        for scale in SCALES:

            resized_pattern = cv2.resize(
                rotated_pattern,
                None,
                fx=scale,
                fy=scale
            )

            h, w = resized_pattern.shape

            # Skip patterns too small or larger than drawing
            if h < 10 or w < 10:
                continue
            if h > H or w > W:
                continue

            # Template matching (normalized cross-correlation)
            result = cv2.matchTemplate(
                drawing,
                resized_pattern,
                cv2.TM_CCOEFF_NORMED
            )

            # Collect all locations above threshold
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                x, y = pt
                score = result[y, x]
                boxes.append([x, y, w, h])
                scores.append(float(score))

    # =========================================================
    # NON-MAXIMUM SUPPRESSION
    # =========================================================

    detections = []
    final_count = 0

    if len(boxes) > 0:

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=threshold,
            nms_threshold=0.3
        )

        # =====================================================
        # DRAW RESULTS
        # =====================================================

        if len(indices) > 0:

            for i in indices.flatten():

                x, y, w, h = boxes[i]
                score = scores[i]

                # Red bounding box
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                # Confidence score label
                cv2.putText(
                    output,
                    f"{score:.2f}",
                    (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                detections.append({
                    "id": final_count + 1,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": round(float(score), 4)
                })

                final_count += 1

    # =========================================================
    # CONVERT OUTPUT TO RGB FOR GRADIO
    # =========================================================

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # =========================================================
    # JSON OUTPUT
    # =========================================================

    json_output = json.dumps(detections, indent=2)

    # =========================================================
    # SUMMARY TEXT
    # =========================================================

    if final_count == 0:
        summary = "❌ No matches found. Try lowering the threshold or adjusting the expected width."
    else:
        avg_conf = sum(d["confidence"] for d in detections) / final_count
        summary = (
            f"✅ Found **{final_count}** match(es)\n"
            f"📐 Drawing size: {W}×{H} px\n"
            f"🔍 Pattern size: {w0}×{h0} px\n"
            f"📊 Avg confidence: {avg_conf:.3f}"
        )

    return output_rgb, json_output, summary


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Pattern Detection Demo",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🔍 Pattern Detection System
    
    Detect pattern occurrences in engineering drawings using **multi-scale**, **rotation-aware** template matching.
    
    **How to use:**
    1. Upload a **Pattern Image** (the symbol or shape to find)
    2. Upload a **Drawing Image** (the full drawing to search within)
    3. Adjust parameters if needed
    4. Click **Run Detection**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pattern_input = gr.Image(label="📌 Pattern Image", type="numpy")
            drawing_input = gr.Image(label="📄 Drawing Image", type="numpy")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                expected_width = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=60,
                    step=5,
                    label="Expected Pattern Width (px)",
                    info="Approximate width of the pattern as it appears in the drawing"
                )
                threshold = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum similarity score to count as a match (higher = stricter)"
                )

            run_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")

        with gr.Column(scale=2):
            output_image = gr.Image(label="🖼️ Detection Result")
            summary_output = gr.Markdown(label="Summary")
            json_output = gr.Code(label="📋 Bounding Boxes (JSON)", language="json")

    # =========================================================
    # EXAMPLE INPUTS
    # =========================================================

    example_dir = "examples"
    if os.path.exists(example_dir):
        example_files = []
        for name in ["example1", "example2", "example3"]:
            p = os.path.join(example_dir, f"{name}_pattern.jpg")
            d = os.path.join(example_dir, f"{name}_drawing.jpg")
            if os.path.exists(p) and os.path.exists(d):
                example_files.append([p, d, 60, 0.8])

        if example_files:
            gr.Examples(
                examples=example_files,
                inputs=[pattern_input, drawing_input, expected_width, threshold],
                label="📂 Example Test Cases"
            )

    # =========================================================
    # EVENT HANDLER
    # =========================================================

    run_btn.click(
        fn=detect,
        inputs=[pattern_input, drawing_input, expected_width, threshold],
        outputs=[output_image, json_output, summary_output]
    )

    gr.Markdown("""
    ---
    **Pipeline:** Grayscale Conversion → Scale Estimation → Multi-Rotation Search (0°/90°/180°/270°) → Multi-Scale Template Matching → Non-Maximum Suppression → Annotated Output
    """)

# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":
    demo.launch(share=True)