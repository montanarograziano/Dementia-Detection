import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# Load your model
model = keras.models.load_model("model/pet.h5")


def predict(image):
    if image is None:
        return "Upload a PET scan first."
    image = preprocess_image(image)
    prediction = round(model.predict(image)[0][0], 2)
    return round(prediction, 2)


def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype="uint8") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def about_section():
    st.title("About MEMENTO Model")
    st.markdown(
        """
    MEMENTO is a family of classification models designed for Alzheimer's Disease detection based on amyloid PET scans and MRI.
    It utilizes advanced machine learning techniques to analyze PET scan images and predict the probability of Alzheimer's Disease.
    At the current stage, two models have been developed:

    - _MEMENTO-PET_: A model that uses only PET scan images for prediction.
    - _MEMENTO-Multimodal_: A model that uses both PET scan and MRI images for prediction.

    Both models have been published, on _SSCI_ 2021 and _Scientific Reports_ 2024 respectively, and you can read more about them in the official papers.

    - [Link to MEMENTO-PET paper](https://ieeexplore.ieee.org/abstract/document/9660102)
    - [Link to MEMENTO-Multimodal paper](https://www.nature.com/articles/s41598-024-56001-9)

    Feel free to reach out for any inquiries or collaborations.
    """,
        unsafe_allow_html=True,
    )


def github_section():
    st.title("GitHub Repository")
    st.markdown(
        """
    The code for MEMENTO has been developed in multiple repositories, due to different stages of development and different experiments.

    You can find the code for with notebooks and experiments in the [this](https://github.com/montanarograziano/Multimodal-approach-for-AD) repository.

    There's also a separate repository (that contains this Streamlit demo) used to develop a FastAPI app for the model, which can be found [here](https://github.com/montanarograziano/Dementia-Detection).
    """,
        unsafe_allow_html=True,
    )


def main():
    st.sidebar.title("**MEMENTO**")
    selected_page = st.sidebar.radio(
        "Navigation", ["Try the tool!", "About", "Code"], label_visibility="hidden"
    )

    if selected_page == "Try the tool!":
        st.title("Alzheimer's Disease Detection - MEMENTO Model")
        st.write("Try the classification model by uploading a PET scan image.")
        st.write("At the moment, only _MEMENTO-PET_ is available.")
        uploaded_file = st.file_uploader(
            "Choose a PET scan image", type=["jpg", "jpeg"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded PET Scan", use_column_width=True)

            # Perform prediction
            prediction = predict(image)

            # Display the prediction result
            st.subheader("Prediction:")
            st.write(f"The predicted probability of is: {prediction}")

    elif selected_page == "About":
        about_section()

    elif selected_page == "Code":
        github_section()


if __name__ == "__main__":
    main()
