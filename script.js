const newsForm = document.getElementById("newsForm");
const newsText = document.getElementById("newsText");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");

const charCount = document.getElementById("charCount");
const loadingPanel = document.getElementById("loadingPanel");
const resultPanel = document.getElementById("resultPanel");
const errorPanel = document.getElementById("errorPanel");

const resultTitle = document.getElementById("resultTitle");
const resultIcon = document.getElementById("resultIcon");
const resultDescription = document.getElementById("resultDescription");
const modelName = document.getElementById("modelName");


function updateCharacterCount() {
    charCount.textContent =
        `${newsText.value.length.toLocaleString()} / 50,000 characters`;
}


function hidePreviousResults() {
    resultPanel.hidden = true;
    errorPanel.hidden = true;
    resultPanel.classList.remove("is-fake");
}


function showError(message) {
    errorPanel.textContent = message;
    errorPanel.hidden = false;
}


function displayResult(data) {
    const prediction = String(data.prediction || "Unknown");
    const normalizedPrediction = prediction.toLowerCase();

    const isFake = normalizedPrediction.includes("fake");
    const isReal = normalizedPrediction.includes("real");

    resultPanel.classList.toggle("is-fake", isFake);

    resultTitle.textContent = prediction;
    resultIcon.textContent = isFake ? "!" : (isReal ? "✓" : "•");

    if (isFake) {
        resultDescription.textContent =
            "The model classified this article as Fake News based on learned text patterns.";
    } else if (isReal) {
        resultDescription.textContent =
            "The model classified this article as Real News based on learned text patterns.";
    } else {
        resultDescription.textContent =
            "The model returned a class that has not been mapped to a known news category. Check the label mapping in the backend.";
    }

    modelName.textContent = `Model: ${data.model || "Logistic Regression"}`;
    resultPanel.hidden = false;

    resultPanel.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


newsText.addEventListener("input", updateCharacterCount);


clearButton.addEventListener("click", () => {
    newsText.value = "";
    updateCharacterCount();
    hidePreviousResults();
    newsText.focus();
});


newsForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const article = newsText.value.trim();

    hidePreviousResults();

    if (!article) {
        showError("Please enter a news article before analyzing.");
        newsText.focus();
        return;
    }

    if (article.length > 50000) {
        showError("Your article exceeds the 50,000-character limit.");
        return;
    }

    analyzeButton.disabled = true;
    loadingPanel.hidden = false;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: article })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error("The server returned an invalid response.");
        }

        if (!response.ok) {
            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "Prediction failed. Please try again.";

            throw new Error(message);
        }

        displayResult(data);

    } catch (error) {
        showError(
            error.message ||
            "Unable to connect to the server. Check that FastAPI is running."
        );
    } finally {
        loadingPanel.hidden = true;
        analyzeButton.disabled = false;
    }
});


updateCharacterCount();