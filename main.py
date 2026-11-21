
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.preprocessing import preprocess_text



BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
MODEL_DIR = BASE_DIR / "models"

# Use the saved Logistic Regression classifier and its vectorizer.
MODEL_PATH = MODEL_DIR / "lr_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Classifier not found: {MODEL_PATH}"
    )

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        f"TF-IDF vectorizer not found: {VECTORIZER_PATH}"
    )

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


vectorizer_features = len(tfidf.vocabulary_)
model_features = getattr(model, "n_features_in_", None)

if model_features is not None and model_features != vectorizer_features:
    raise ValueError(
        "Model-vectorizer mismatch: "
        f"classifier expects {model_features} features, "
        f"but vectorizer provides {vectorizer_features}. "
        "Load the classifier and vectorizer saved from the same pipeline."
    )

print(f"Loaded classifier: {MODEL_PATH.name}")
print(f"Loaded vectorizer: {VECTORIZER_PATH.name}")
print(f"Vectorizer vocabulary size: {vectorizer_features}")
print(f"Model classes: {model.classes_}")



# Verify these meanings against your dataset documentation.
# Based on the examples previously inspected, 0 appears to be Fake
# and 1 appears to be Real, but this must be confirmed.
LABEL_MAP = {
    "0": "Fake News",
    "1": "Real News",
    "fake": "Fake News",
    "real": "Real News",
}



app = FastAPI(
    title="Is it real or fake",
    description="News text classification using machine learning.",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


class NewsRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=50000,
        description="News article text to classify",
    )


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "classifier_file": MODEL_PATH.name,
        "vectorizer_file": VECTORIZER_PATH.name,
        "vocabulary_size": vectorizer_features,
    }


@app.post("/predict")
def predict_news(request: NewsRequest):
    article = request.text.strip()

    if not article:
        raise HTTPException(
            status_code=400,
            detail="Please enter a news article.",
        )

    try:
        # Apply the same preprocessing procedure used during training.
        processed_text = preprocess_text(article)

        if not processed_text.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable text found. "
                    "Please enter a longer article."
                ),
            )

        # Transform with the saved, fitted vectorizer.
        features = tfidf.transform([processed_text])

        # Predict the numeric class.
        prediction = model.predict(features)[0]
        raw_label = str(prediction)

        # Convert the numeric label into a readable result.
        result_label = LABEL_MAP.get(
            raw_label.lower(),
            f"Unknown class ({raw_label})",
        )

        response = {
            "prediction": result_label,
            "raw_label": raw_label,
            "model": "Logistic Regression",
            "message": (
                "This prediction reflects patterns learned from the "
                "training data. It does not independently verify "
                "the article's claims."
            ),
        }

        # Include probabilities if the classifier supports them.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            class_scores = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            }

            predicted_score = next(
                (
                    float(prob)
                    for cls, prob in zip(classes, probabilities)
                    if str(cls) == raw_label
                ),
                None,
            )

            response["class_scores"] = class_scores

            if predicted_score is not None:
                response["predicted_class_score"] = round(
                    predicted_score, 4
                )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from exc


# uvicorn app.main:app --reload