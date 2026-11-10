# Heart Sound Classification

A Streamlit application that classifies an uploaded heart-sound recording into one of three model output classes: **Normal**, **Murmur**, or **Extrasystole**.

> **Research/educational project only.** This model is not a medical device and must not be used to diagnose, rule out, or make treatment decisions about a heart condition. Predictions require clinical interpretation by a qualified professional.

## Repository contents

| File | Purpose |
|---|---|
| `heartbeatdetection.ipynb` | Model development and training notebook |
| `heartbeat_model.h5` | Saved TensorFlow/Keras model used by the app |
| `heartbeat_model.pkl` | Existing serialized artifact; its role has not yet been verified |
| `app (1).py` | Streamlit inference interface |
| `requirements.txt` | Python dependencies |

## How the application works

1. Upload a heart-sound recording in WAV format.
2. Load and resample the audio to 22,050 Hz.
3. Extract 25 Mel-frequency cepstral coefficients (MFCCs).
4. Pad or truncate the feature sequence to 500 frames.
5. Add a batch dimension and pass the tensor to the saved Keras model.
6. Display the predicted class and the model's class scores.

Current inference tensor layout: `(batch, 25, 500)`.

## Run locally

Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run "app (1).py"
```

Keep `heartbeat_model.h5` in the same directory as the Streamlit app.

## Important model consistency note

The current application code uses 5 seconds of audio, 22,050 Hz, 25 MFCC coefficients, and 500 frames. These settings must exactly match the preprocessing used to train the committed model. The training notebook's content could not be inspected from the repository file endpoint during this documentation pass, so training/inference parity and the class-index order have **not** been independently verified. The app checks the loaded model's input/output dimensions at runtime, but matching dimensions alone do not prove matching preprocessing or label semantics.

Before reporting performance, evaluate on a held-out, labelled test set that was not used for training, validation, threshold selection, or tuning. Report per-class precision/recall/F1, macro-F1, a confusion matrix, and the test-set class distribution. Do not describe accuracy measured on unlabelled or training data as test accuracy.

## Limitations

- Predictions depend on recording quality, segmentation, dataset coverage, and the model's training distribution.
- The displayed scores are model outputs, not calibrated clinical probabilities.
- The current three-class label mapping must be confirmed against the label encoder/class order used during training.
- The purpose of `heartbeat_model.pkl` remains undocumented until its contents and use are verified.

## Suggested next validation steps

- Confirm the notebook's sample rate, audio duration/windowing, MFCC parameters, tensor layout, and class-index mapping.
- Re-train or export the model with the exact preprocessing pipeline used by the app if they differ.
- Save the best validation checkpoint rather than automatically using the final epoch.
- Evaluate once on an untouched, labelled test set and document the resulting metrics.
