import io
import os

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_PATH = "heartbeat_model.h5"
SAMPLE_RATE = 22_050
AUDIO_DURATION = 5
N_MFCC = 25
MAX_FRAMES = 500

# IMPORTANT: Confirm this order against the encoder used during training.
CLASS_NAMES = ["normal", "murmur", "extrahlsystole"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. "
            "Place it in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes into the MFCC tensor expected by the current app."""
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        duration=AUDIO_DURATION,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The uploaded audio contains no readable samples.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
    )

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])),
            mode="constant",
        )
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    return np.expand_dims(mfcc.astype(np.float32), axis=0)


def main():
    st.set_page_config(
        page_title="Heart Sound Classification",
        page_icon="💓",
        layout="centered",
    )

    st.title("💓 Heart Sound Classification")
    st.write(
        "Upload a WAV recording to obtain a model prediction "
        "for the supported heart-sound classes."
    )

    st.warning(
        "Educational/research use only. This tool is not a medical device "
        "and cannot diagnose or rule out a heart condition. Consult a "
        "qualified healthcare professional for medical concerns."
    )

    uploaded_file = st.file_uploader(
        "Upload a heart-sound recording",
        type=["wav"],
        help="Use a clear WAV recording for best input quality.",
    )

    if uploaded_file is None:
        st.info("Upload a WAV file to begin.")
        return

    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Classify heart sound", type="primary"):
        try:
            model = load_model()
            model_input = preprocess_audio(audio_bytes)

            # Validate input dimensions instead of silently reshaping features.
            expected_shape = model.input_shape
            if isinstance(expected_shape, list):
                raise ValueError("This app currently supports single-input models only.")

            actual_shape = tuple(model_input.shape[1:])
            expected_feature_shape = tuple(
                dim if dim is not None else actual
                for dim, actual in zip(expected_shape[1:], actual_shape)
            )

            if len(expected_shape) != len(model_input.shape) or actual_shape != expected_feature_shape:
                raise ValueError(
                    f"Preprocessing produced {actual_shape}, but the model "
                    f"expects {expected_shape[1:]}. Check the training notebook "
                    "and use exactly the same sample rate, duration, MFCC "
                    "settings, and tensor layout."
                )

            scores = np.asarray(model.predict(model_input, verbose=0))

            if scores.ndim != 2 or scores.shape[0] != 1:
                raise ValueError(f"Unexpected model output shape: {scores.shape}")

            if scores.shape[1] != len(CLASS_NAMES):
                raise ValueError(
                    f"The model returned {scores.shape[1]} outputs, but "
                    f"CLASS_NAMES contains {len(CLASS_NAMES)} labels. "
                    "Verify the training label mapping."
                )

            if not np.all(np.isfinite(scores)):
                raise ValueError("The model returned non-finite prediction values.")

            predicted_index = int(np.argmax(scores[0]))
            st.success(f"Predicted class: **{CLASS_NAMES[predicted_index]}**")

            st.subheader("Model output scores")
            score_table = {
                "Class": CLASS_NAMES,
                "Score": [float(value) for value in scores[0]],
            }
            st.dataframe(score_table, hide_index=True, use_container_width=True)

            st.caption(
                "Scores are raw model outputs and are not calibrated clinical "
                "probabilities. They should not be interpreted as a diagnosis."
            )

        except Exception as error:
            st.error(f"Could not classify this recording: {error}")


if __name__ == "__main__":
    main()
