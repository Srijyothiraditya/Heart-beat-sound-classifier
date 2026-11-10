# Heart Sound Classification

A Streamlit application that classifies an uploaded heart-sound recording into one of three model output classes: **Normal**, **Murmur**, or **Extrasystole**.

> **Research/educational project only.** This model is not a medical device and must not be used to diagnose, rule out, or make treatment decisions about a heart condition. Predictions require clinical interpretation by a qualified professional.

## Repository files

| File | Purpose |
|---|---|
| `heartbeatdetection.ipynb` | Model development and training notebook |
| `heartbeat_model.h5` | Saved TensorFlow/Keras model used by the app |
| `heartbeat_model.pkl` | Existing serialized artifact; its role still needs verification |
| `app.py` | Streamlit inference application |
| `app (1).py` | Earlier application version |
| `requirements.txt` | Python dependencies |

## Inference workflow

1. Upload a WAV heart-sound recording.
2. Load it as mono audio and resample to 22,050 Hz.
3. Use the first 5 seconds of audio.
4. Extract 25 MFCC coefficients.
5. Pad or truncate the feature sequence to 500 frames.
6. Add a batch dimension and pass the tensor to the saved Keras model.
7. Display the predicted class and raw model output scores.

The current app produces an input tensor with shape `(1, 25, 500)`. It checks the loaded model's expected input shape and output count before displaying a prediction.

## Run locally

Use a Python environment compatible with the TensorFlow version installed on your system.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Keep `heartbeat_model.h5` in the same directory as `app.py`.

## Important: training and inference must match

The current app uses 5 seconds, 22,050 Hz, 25 MFCC coefficients, 500 frames, and the tensor layout `(batch, 25, 500)`. These values must exactly match the preprocessing used to train the committed model. The training notebook could not be inspected during this update, so preprocessing parity and the class-index order have **not** been independently verified. Matching tensor dimensions alone does not prove that the features or labels are semantically correct.

The app currently assumes the output order is `normal`, `murmur`, `extrahlsystole`. Confirm this against the exact label encoder or class-index mapping used during training before relying on predictions.

## Evaluation checklist

Evaluate only on an untouched, labelled test set that was not used for training, validation, model selection, or threshold tuning. Report:
- Per-class precision, recall, and F1-score
- Macro-F1 and accuracy
- Confusion matrix
- Number of samples per class
- Test split strategy and any patient/recording-level grouping used to prevent leakage

Do not report training accuracy as test performance. If recordings from the same patient or original recording appear in multiple splits, the evaluation may be overly optimistic.

## Limitations

- Recording quality, background noise, segmentation, and dataset coverage can affect predictions.
- Raw model scores are not calibrated clinical probabilities.
- This project is for learning and research, not clinical diagnosis or treatment decisions.
- The purpose of `heartbeat_model.pkl` must be documented after its contents and use are verified.

## Remaining verification

1. Inspect the training notebook and confirm sample rate, duration, MFCC configuration, padding/truncation, tensor layout, and label order.
2. Confirm the model's actual input/output shapes from `heartbeat_model.h5`.
3. Confirm what `heartbeat_model.pkl` stores and whether the application needs it.
4. Run the app with known WAV examples and check that predictions execute end-to-end.
5. Evaluate on a separate labelled test set and document measured results.
