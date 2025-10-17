import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Gamma vs Hadron Classifier",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom Styling --------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #00FFFF;
            font-size: 38px;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            color: #A0A0A0;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #1f1f1f;  
            color:#FFFFFF;             
            border: 2px solid #b34747;  
            border-radius: 10px;        
            padding: 10px 24px;         
            font-size: 16px;
            font-weight: bold;
            display: inline-block;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #D2042D;
        }
        .result-box {
            background-color: #111;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 class="main-title">
    <span style="color:#00FFFF;">Gamma</span> 
    <span style="color:#F9F6EE;">Vs</span> 
    <span style="color:#D2042D;">Hadron</span> 
</h1>
""", unsafe_allow_html=True)


model = joblib.load("MAGIC_model.pkl")
scaler = joblib.load("MAGIC_scaler.pkl")


st.sidebar.header("Input Particle Features")
st.sidebar.write("Adjust the sliders or type values manually:")

fLength = st.sidebar.slider("fLength", 0.0, 200.0, 30.0)
fWidth = st.sidebar.slider("fWidth", 0.0, 200.0, 15.0)
fSize = st.sidebar.slider("fSize", 0.0, 5000.0, 1500.0)
fConc = st.sidebar.slider("fConc", 0.0, 1.0, 0.3)
fConc1 = st.sidebar.slider("fConc1", 0.0, 1.0, 0.1)
fAsym = st.sidebar.slider("fAsym", 0.0, 20.0, 5.0)
fM3Long = st.sidebar.slider("fM3Long", 0.0, 100.0, 50.0)
fM3Trans = st.sidebar.slider("fM3Trans", 0.0, 100.0, 10.0)
fAlpha = st.sidebar.slider("fAlpha", 0.0, 1.0, 0.1)
fDist = st.sidebar.slider("fDist", 0.0, 400.0, 200.0)

features = np.array([[fLength, fWidth, fSize, fConc, fConc1, fAsym,
                      fM3Long, fM3Trans, fAlpha, fDist]])

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    st.write("")
    predict_button = st.button("Predict Event",use_container_width=True)





if predict_button:
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        result = int(prediction[0])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Prediction Result: ")

        if result == 1:
            st.success("**Gamma Ray Particle Detected!**")
            st.markdown(
                "<div class='result-box'><h3 style='color:#00FF7F;'>G.A.M.M.A Event Detected</h3></div>",
                unsafe_allow_html=True
            )
        else:
            st.error("**Hadron Particle Detected!**")
            st.markdown(
                "<div class='result-box'><h3 style='color:#FF6347;'>H.A.D.R.O.N Event Detected</h3></div>",
                unsafe_allow_html=True
            )


    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Batch Prediction (Upload CSV File)")
st.write("Upload a `CSV` file containing particle feature values to classify multiple entries at once.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                         "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]

        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded CSV must contain the required feature columns.")
        else:
            st.write("✅ File Uploaded Successfully! Preview:")
            st.dataframe(df.head())

            scaled_data = scaler.transform(df[required_cols])
            predictions = model.predict(scaled_data)
            df["Prediction"] = ["Gamma (1)" if p == 1 else "Hadron (0)" for p in predictions]

            st.success("Batch Prediction Completed!")
            st.dataframe(df)

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error reading or processing file: {e}")


with st.expander("Documentation / Instructions"):
    st.markdown("""
    ### Input Features
    - `fLength`: Length of the particle image (0-200)
    - `fWidth`: Width of the particle image (0-200)
    - `fSize`: Size parameter (0-5000)
    - `fConc`, `fConc1`: Concentration parameters (0-1)
    - `fAsym`: Asymmetry parameter (0-20)
    - `fM3Long`, `fM3Trans`: Third moment parameters (0-100)
    - `fAlpha`: Angle parameter (0-1)
    - `fDist`: Distance parameter (0-400)
    
    ### How to Use
    1. Adjust the sliders in the sidebar for single particle prediction.
    2. Press **Predict Particle Type** to see the classification.
    3. For batch prediction, upload a CSV containing the above features.
    4. Download the results as a CSV for further analysis.
    """)

# -------------------- Model Evaluation Documentation --------------------
with st.expander("Model Evaluation Metrics"):
    st.markdown("""
    ## Model Evaluation Metrics
                
    Model: `Gahon-v4.1.0`
                
    The **Gamma vs Hadron Classifier** was evaluated on a test dataset with **5,706 samples**. Below are the key performance metrics:

    ### Accuracy
    - **Overall Accuracy:** `88.03%`
      This means the model correctly classifies approximately **88 out of 100 particles**.

    ### Classification Report

    | Class | Precision | Recall | F1-Score | Support |
    |-------|----------|--------|----------|--------|
    | `Hadron` (0) | 0.87 | 0.77 | 0.82 | 2,001 |
    | `Gamma` (1)  | 0.88 | 0.94 | 0.91 | 3,705 |

    - **Macro Average**  
      - Precision: 0.88  
      - Recall: 0.86  
      - F1-Score: 0.86  
                

    - **Weighted Average**  
      - Precision: 0.88  
      - Recall: 0.88  
      - F1-Score: 0.88  

    ### Interpretation
    - The model performs **better on Gamma particles (class 1)** than on Hadron (class 0), as seen from higher recall and F1-score.  
    - Precision is balanced for both classes, indicating low false positive rates.  
    - Overall, the classifier demonstrates **robust performance** for identifying particle types.
    """)
