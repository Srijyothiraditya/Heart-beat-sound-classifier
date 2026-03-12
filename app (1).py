import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

# Load the trained model
@st.cache_resource
def load_heartbeat_model():
    model = tf.keras.models.load_model("heartbeat_model.h5")  # replace with your .h5 file
    return model

model = load_heartbeat_model()

# Streamlit UI
st.title("💓 Heartbeat Sound Classification App")
st.write("Upload a heartbeat sound (.wav) file to classify it (Normal / Murmur / Extrasystole / etc.)")

uploaded_file = st.file_uploader("Choose a heartbeat audio file", type=["wav"])

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load audio
        y, sr = librosa.load(tmp_path, duration=5, sr=22050)

        # Extract MFCCs (25 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)  # shape: (25, t)

        # Pad or truncate to 500 frames (matches model input)
        max_len = 500
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        # DO NOT transpose
        # Shape now: (25, 500)
        X = np.expand_dims(mfcc, axis=0)  # Add batch dimension -> (1, 25, 500)
        st.write(f"Input shape to model: {X.shape}")  # Debugging

        # Make prediction
        prediction_probs = model.predict(X)
        predicted_class = np.argmax(prediction_probs, axis=1)[0]

        # Map numeric labels to class names
        class_names = ['normal', 'murmur', 'extrahlsystole']  # update if needed
        st.success(f"### 🩺 Predicted Heartbeat Type: **{class_names[predicted_class]}**")

        # Show confidence scores
        st.write("#### Confidence Scores:")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {prediction_probs[0][i]:.4f}")

        # Optional: play audio
        st.audio(uploaded_file)

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
