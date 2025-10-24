# CAPTCHA Digit Solver 🔢🤖

This project demonstrates how to build a deep learning model using YOLOv8 to automatically detect and recognize digits in CAPTCHA images.
It highlights skills in data preparation, model training, evaluation, and inference deployment.

---

## 🚀 Project Highlights
- Built a custom dataset of CAPTCHA images containing digits only.
- Manually annotated all digits using Roboflow (YOLO format).
- Trained a YOLOv8 object detection model for digit recognition.
- Evaluated performance using precision, recall, and F1 metrics.
- Deployed a Python-based inference pipeline for real-time digit extraction.
 
---

## 🧠 Tech Stack

| Component        | Tool                                                 |
| ---------------- | ---------------------------------------------------- |
| Object Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) |
| Data Labeling    | Roboflow                                             |
| Framework        | PyTorch                                              |
| Environment      | Python 3.10+                                         |
| Visualization    | Matplotlib, OpenCV                                   |

---

## 📂 Project Structure

---

## ⚙️ Setup

---

## 🏋️‍♂️ Training
```bash
python src/train.py --data data/data.yaml --epochs 50 --img 640
```

---

## 🔍 Inference Example

```bash
python run.py --image_path sample_captcha.png
```

Output example:
Detected digits: [4, 9, 3, 1, 7]

---

## 📊 Evaluation

Precision-Recall, F1-Score, and Confusion Matrix visualizations are available in:
```bash
model/metrics/
```
---

## 💡 Future Improvements

- Add character (A-Z) recognition for alphanumeric CAPTCHAs.
- Build a web app demo (Streamlit or Gradio).
- Integrate OCR post-processing for full CAPTCHA solving.

---

## 🧾 License

MIT License © 2025 Israa Elhouch

---
## 🧑‍💻 Author

Israa Elhouch — AI & ML Developer
LinkedIn
 • GitHub