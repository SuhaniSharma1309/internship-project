import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector import process_image, process_video, set_zone
import os
import tempfile
import pandas as pd
import plotly.express as px

# define style variable to avoid NameError if referenced elsewhere
style = ""

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Air & Ground Surveillance System",
    page_icon="🚁",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #000000;
    background-image:
        linear-gradient(rgba(0,255,100,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,100,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
}
[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-weight: bold;
}
h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;'>
AI-Powered Air & Ground Surveillance and Threat Assessment System</h1>
<h5 style='text-align:center; color:gray;'>
Internship Research Project | Computer Vision Based Threat Detection System
</h5>
""", unsafe_allow_html=True)

# Banner slides

st.markdown("""

<style>
div.stButton > button {
    background-color: #00aa55;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}
div.stButton > button:hover {
    background-color: #00cc66;
}
</style>
""", unsafe_allow_html=True)


slides = [
    "assets/banner.png",
    "assets/banner1.png",
    "assets/banner2.png"
]

st.markdown("""
<style>

/* File uploader box */
[data-testid="stFileUploader"] {
    background-color: #111111;
    border: 2px solid #00aa55;
    border-radius: 10px;
    padding: 10px;
}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background-color: #00aa55 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: bold !important;
}

/* Hover effect */
[data-testid="stFileUploader"] button:hover {
    background-color: #00cc66 !important;
}

</style>
""", unsafe_allow_html=True)

# Session state

if "banner_index" not in st.session_state:
    st.session_state.banner_index = 0

# Navigation buttons

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    if st.button("⬅"):
        st.session_state.banner_index = (
            st.session_state.banner_index - 1
        ) % len(slides)

with col3:
    if st.button("➡"):
        st.session_state.banner_index = (
            st.session_state.banner_index + 1
        ) % len(slides)

idx = st.session_state.banner_index

# Display banner

st.image(
    slides[idx],
    use_container_width=True
)


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------

for key, val in [("objects", 0), ("threats", 0), ("det", []), ("result_img", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

metrics_placeholder = st.empty()

def render_metrics():
    with metrics_placeholder.container():
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Detected Objects", st.session_state.objects)
        with k2: st.metric("Threats", st.session_state.threats)
        with k3: st.metric("Restricted Zone", "ACTIVE" if st.session_state.threats > 0 else "CLEAR")
        with k4: st.metric("System Status", "ONLINE")

# --------------------------------------------------
# CONTROLS
# --------------------------------------------------

from streamlit_image_select import image_select

st.subheader("🖼️ Select Analysis Mode")

analysis_mode = image_select(
    label=" ",
    images=[
        "assets/img_mode.png",
        "assets/video_mode.png"
    ],
    captions=[
        "Image Analysis",
        "Video Analysis"
    ]
)
st.subheader("🖼️ Select Surveillance Type")

surveillance_type = image_select(
    label=" ",
    images=[
        "assets/air_surveillance.png",
        "assets/ground_surveillance.png"
    ],
    captions=[
        "Air Surveillance",
        "Ground Surveillance"
    ]
)
# Analysis mode

if analysis_mode == "assets/img_mode.png":
    mode = "Image Analysis"
else:
    mode = "Video Analysis"

# Surveillance type

if surveillance_type == "assets/air_surveillance.png":
    type_ = "Air Surveillance"
else:
    type_ = "Ground Surveillance"
    
conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

if mode == "Image Analysis":
    render_metrics()
    st.divider()

# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------

if mode == "Image Analysis":

    uploaded_file = st.file_uploader(
        "📸 Upload Surveillance Image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file:
        img    = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.info("Set restricted zone coordinates (defaults to center 40% of image)")
        h, w = img_np.shape[:2]
        zi1, zi2, zi3, zi4 = st.columns(4)
        with zi1: x1 = st.number_input("Zone X1", 0, w, int(w * 0.3))
        with zi2: y1 = st.number_input("Zone Y1", 0, h, int(h * 0.3))
        with zi3: x2 = st.number_input("Zone X2", 0, w, int(w * 0.7))
        with zi4: y2 = st.number_input("Zone Y2", 0, h, int(h * 0.7))

        set_zone(x1, y1, x2, y2)

        if st.button("🚀  Run Detection", width=200):
            with st.spinner("Running detection..."):
                result, det = process_image(img_np.copy(), mode=type_, conf=conf)
            st.session_state.det = det
            st.session_state.result_img = result
            st.session_state.objects = len(det)
            st.session_state.threats = sum(1 for d in det if d["threat"] == "HIGH RISK")
            render_metrics()

        if st.session_state.result_img is not None:
            col1, col2, col3 = st.columns([1,3,1])

            with col2:
                st.image(
                    st.session_state.result_img,
                    caption="Detection Output",
                    width=700
                )
    
        if st.session_state.det:
            df = pd.DataFrame(st.session_state.det)

            tab1, tab2, tab3 = st.tabs([
                "📋 Detection Table",
                "⚠️ Threat Analysis",
                "📊 Object Distribution"
            ])

            # -----------------------------
            # TAB 1 - TABLE
            # -----------------------------
            with tab1:
                st.dataframe(
                    df,
                    use_container_width=True
                )

            # -----------------------------
            # TAB 2 - THREAT ANALYSIS
            # -----------------------------
            with tab2:

                threat_counts = (
                    df["threat"]
                    .value_counts()
                    .reset_index()
                )

                threat_counts.columns = [
                    "Threat Level",
                    "Count"
                ]

                st.bar_chart(
                    threat_counts.set_index(
                        "Threat Level"
                    )
                )

                st.metric(
                    "Total High Risk Threats",
                    sum(df["threat"] == "HIGH RISK")
                )
                
            # -----------------------------
            # TAB 3 - OBJECT ANALYSIS
            # -----------------------------
            with tab3:

                object_counts = (
                    df["label"]
                    .value_counts()
                    .reset_index()
                )

                object_counts.columns = [
                    "Object",
                    "Count"
                ]


                fig = px.bar(
                        object_counts,
                        x="Object",
                        y="Count",
                        title="Detected Object Distribution",
                        color="Object",
                        color_discrete_sequence=[
                            "#ffaa00",
                            "#00ccff",
                            "#00ff88",
                            "#ff4444",
                            "#aa66ff"
                    ]
                )

                fig.update_layout(
                    height=350,      # decrease height
                    width=600,        # optional width
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font_color="white"
                )

                st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# VIDEO MODE
# --------------------------------------------------

else:
    uploaded_file = st.file_uploader(
        "🎥 Upload Surveillance Video",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_file:
        tmp_dir    = tempfile.gettempdir()
        input_path = os.path.join(tmp_dir, "input_video.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("🚀  Run Video Detection", width=200):

            with st.spinner("Processing video — this may take a moment..."):
                processed_path, total_objects, total_threats = process_video(
                    input_path, mode=type_, conf=conf
                )

            if not os.path.exists(processed_path):
                st.error("Processing failed — output file not found.")
            elif os.path.getsize(processed_path) == 0:
                st.error("Processing failed — output file is empty.")
            else:
                st.session_state.objects = total_objects
                st.session_state.threats = total_threats

                st.subheader("Detection Output")
                col1, col2, col3 = st.columns([3,4,3])

                with col2:
                    with open(processed_path, "rb") as f:
                        st.video(f.read())

                st.success(
                    f"Output: {os.path.getsize(processed_path) / 1024:.1f} KB"
                )

        st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray'>
Developed as an Internship Research Project <br>
AI-Based Air & Ground Surveillance System
</div>
""", unsafe_allow_html=True)
