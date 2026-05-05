# 🏙️ Smart Office Monitor

A real-time, edge-computing desk monitoring system that combines YOLO-based pose estimation, object detection, and DeepFace facial recognition to track employee presence and activity.

## 🚀 Features

- **Real-time Occupancy Tracking**: Automatically identifies desks and tracks their occupancy status.
- **Identity Recognition**: Uses DeepFace (ArcFace) to distinguish between enrolled employees and guests.
- **Activity Classification**: Detects behaviors like **Working**, **Sitting Idle**, **Walking**, **Standing**, and **Using Mobile**.
- **Interactive Dashboard**: A modern Streamlit interface providing live updates and historical activity breakdowns.
- **Automatic Data Logging**: Stores accumulated activity durations per employee.
- **Face Enrollment Utility**: Easy-to-use script for adding new employees to the recognition database.

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, Ultralytics YOLOv8 (Pose & Object detection).
- **Facial Recognition**: DeepFace (ArcFace backend).
- **Backend**: FastAPI (Python).
- **Frontend**: Streamlit.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZaheerAfzal1408/Desk-Monitoring-System.git
   cd "Desk Monitoring System"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO Models**:
   The system uses `yolov8n-pose.pt` and `yolov8n.pt`. These will be downloaded automatically by the script on the first run, or you can place them in the root directory.

---

## 🏃 How to Run

To run the full system, you need to start the backend, the core monitor, and the dashboard.

### 1. Start the Backend API
The backend stores the state and provides data to the dashboard.
```bash
python backend.py
```

### 2. Start the Monitoring System (Main Loop)
This script handles the camera feed, detection, and recognition.
```bash
python main.py
```

### 3. Launch the Dashboard
Open the interactive UI in your browser.
```bash
streamlit run dashboard.py
```

---

## 👤 Employee Enrollment

To add a new employee to the facial recognition database:

1. Run the enrollment script:
   ```bash
   python enroll.py
   ```
2. Enter the employee's name.
3. Use the 'S' key to save photos (take at least 3-5 photos from different angles).
4. Press 'Q' to finish.

The system will automatically detect the new photos and rebuild the recognition cache on the next run of `main.py`.

---

## 📁 Project Structure

- `main.py`: Core logic for computer vision and state reporting.
- `backend.py`: FastAPI server for state management.
- `dashboard.py`: Streamlit dashboard for visualization.
- `enroll.py`: Utility for employee face enrollment.
- `employees/`: Directory containing employee photo databases.
- `requirements.txt`: Python dependencies.

---

## ⚠️ Requirements
- Python 3.9+
- Webcam (or IP camera stream configured in `main.py`)
- CUDA-compatible GPU (Optional, but recommended for better FPS)
