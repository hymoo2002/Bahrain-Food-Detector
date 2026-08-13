import random
from datetime import datetime

import streamlit as st




st.set_page_config(
    page_title="Traditional Food Recognizer",
    page_icon="🍽️",
    layout="centered",
)

FOOD_CLASSES = [
    "Balaleet",
    "Fried Fish",
    "Luqaimat",
    "Samboosa",
    "Machboos",
    "Harees",
    "Bahraini Halwa",
    "Nakhi",
    "Matai",
    "Kebab",
]



def predict_dish(image):
    """
    Temporary mock prediction.

    Replace this function with your actual
    computer vision model when it is ready.
    """

    scores = [random.random() for _ in FOOD_CLASSES]
    total = sum(scores)

    predictions = [
        {
            "name": name,
            "confidence": score / total,
        }
        for name, score in zip(FOOD_CLASSES, scores)
    ]

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return predictions[:3]


# -----------------------------
# History
# -----------------------------

def save_prediction(predictions):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(
        0,
        {
            "name": predictions[0]["name"],
            "confidence": predictions[0]["confidence"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )


# -----------------------------
# App UI
# -----------------------------

st.title("🍽️ Traditional Food Recognizer")

st.write(
    "Upload a photo or take a photo to identify a traditional dish."
)


source = st.radio(
    "Choose image source",
    ["Upload image", "Camera"],
    horizontal=True,
)


# Image input
image = None

if source == "Upload image":
    image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

else:
    image = st.camera_input("Take a photo")


# Display image and prediction
if image:

    st.image(
        image,
        caption="Selected image",
        use_container_width=True,
    )

    if st.button(
        "🔍 Recognize Dish",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Analyzing image..."):

            predictions = predict_dish(image)

        # Save result
        save_prediction(predictions)

        # Best prediction
        best = predictions[0]

        st.subheader("Prediction")

        st.success(
            f"**{best['name']}**"
        )

        st.metric(
            "Confidence",
            f"{best['confidence']:.1%}",
        )

        # Other predictions
        st.subheader("Other possibilities")

        for prediction in predictions[1:]:

            st.write(
                f"**{prediction['name']}** — "
                f"{prediction['confidence']:.1%}"
            )

else:

    st.info(
        "Upload an image or take a photo to begin."
    )