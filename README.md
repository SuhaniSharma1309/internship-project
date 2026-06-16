# AI-Powered Air & Ground Surveillance and Threat Assessment System
DRDO RESEARCH PROJECT

## Overview

This project is an AI-powered surveillance system designed for monitoring aerial and ground environments using Computer Vision and Deep Learning techniques. The system performs object detection, threat prediction, and restricted zone monitoring on both images and video streams.

The primary focus of the project is image-based surveillance, while video-based surveillance capabilities have also been implemented and will be further enhanced in future versions.

The system supports detection of aerial and ground objects, visualizes detections using bounding boxes, predicts threat levels, and provides detailed detection information for analysis and decision-making.

## Live Demo

### Streamlit Application
🔗 [Launch Application](https://threatdetector.streamlit.app/)

### Demo Video
🎥 [Watch Demo Video](https://drive.google.com/file/d/1x9UE_cRnihb1giU_EIayc_2C38vM8h3J/view?usp=sharing)

---

## Features

### Air Surveillance
- Detection of aerial objects from images and videos
- Threat prediction for detected aerial objects
- Restricted zone monitoring using user-defined coordinates
- Bounding box visualization with confidence scores

### Ground Surveillance
- Detection of ground objects using a dedicated model
- Threat prediction and object classification
- Detection visualization using bounding boxes

### Image Analysis
- Object detection
- Threat prediction
- Restricted zone violation detection
- Detection summary in tabular format
- System status monitoring

### Video Analysis
- Real-time object detection
- Threat prediction
- Detection visualization

### Additional Features
- Adjustable confidence threshold slider
- Detection details table containing:
  - Object Label
  - Confidence Score
  - Threat Status
- User-configurable restricted zone coordinates

---

## Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* OpenCV
* YOLO
* NumPy
* Matplotlib

### Development Tools

* VS Code
* Colab Notebook

---

## Datasets

### Air Surveillance Dataset
- Source: Roboflow
- Purpose: Detection of aerial objects

### Ground Surveillance Dataset
- Source: VisDrone Dataset
- Purpose: Detection of ground objects and surveillance targets

---

## Methodology

The project follows the workflow below:

Dataset Collection -> Data Preprocessing -> Data Annotation -> Model Training -> Object Detection -> Performance Evaluation -> Result Visualization

---

## Models Used

### Air Surveillance Model
- YOLOv8
- Trained for aerial object detection and threat assessment

### Ground Surveillance Model
- YOLOv8 Nano (YOLOv8n)
- Used for ground object detection to ensure faster inference and reduced computational requirements


---

## Future Enhancements

- Dynamic restricted zone selection for video surveillance
- Improved video analytics pipeline
- Multi-object tracking
- Real-time surveillance deployment
- Enhanced threat assessment mechanisms
- Integration with alert and notification systems

### Sample Detection Results


![Detection Result 1](/pic1.png)

---


![Detection Result 2](/pic2.png)

---


![Detection Result 3](/pic3.png)

---


![Detection Result 3](/pic4.png)

---


![Detection Result 3](/pic5.png)

---


![Detection Result 3](/pic6.png)

---
## Acknowledgements

I would like to express my gratitude to my mentor, Mr Ravi Kumar Meena Sir and the team at SSPL-DRDO for their guidance and support throughout the internship.

---

## Author

Suhani Sharma

B.Tech Computer Science Engineering

Summer Intern, SSPL-DRDO

Amity University, Noida
