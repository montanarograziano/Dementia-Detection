import numpy as np
import streamlit as st
from tensorflow import keras

memento_3d_mri_model = keras.models.load_model("model/mri-3d.h5")
memento_3d_pet_model = keras.models.load_model("model/pet-3d.h5")
memento_multimodal_model = keras.models.load_model("model/merged.h5")


# Function to select the appropriate model
def select_model(model_name):
    if model_name == "MEMENTO-3D-MRI":
        return memento_3d_mri_model
    elif model_name == "MEMENTO-3D-PET":
        return memento_3d_pet_model
    elif model_name == "MEMENTO-MULTIMODAL":
        return memento_multimodal_model
    else:
        return None


def predict(image_1, image_2, selected_model):
    if selected_model is None:
        return "Select a model first."

    if selected_model == memento_multimodal_model and (
        image_1 is None or image_2 is None
    ):
        return "Upload both PET and MRI scans for the MEMENTO-MULTIMODAL model."

    if selected_model == memento_multimodal_model:
        image_1 = preprocess_image(image_1)
        image_2 = preprocess_image(image_2)
        prediction = selected_model.predict([image_1, image_2])[0][0]
    else:
        image = preprocess_image(image_1)
        prediction = selected_model.predict(image)[0][0]

    return round(prediction, 3)


def preprocess_image(img_array):
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def download_sample_files():
    pet_sample_path = "data/sample-pet.npy"
    mri_sample_path = "data/sample-mri.npy"

    st.download_button(
        "Download sample PET", data=pet_sample_path, file_name="sample-pet.npy"
    )
    st.download_button(
        "Download sample MRI", data=mri_sample_path, file_name="sample-mri.npy"
    )


def about_section():
    st.title("About MEMENTO Model")
    st.markdown(
        """
    MEMENTO is a family of classification models designed for Alzheimer's Disease detection based on amyloid PET scans and MRI.
    It utilizes advanced machine learning techniques to analyze PET scan images and predict the probability of Alzheimer's Disease.
    At the current stage, the following models have been developed and are available for use in this demo:

    - _MEMENTO-3D-MRI_: A model that uses only 3D MRI images for prediction.
    - _MEMENTO-3D-PET_: A model that uses only 3D PET scan images for prediction.
    - _MEMENTO-Multimodal_: A model that uses both PET scan and MRI images for prediction.

    The last two models have been published, on _SSCI_ 2021 and _Scientific Reports_ 2024 respectively, and you can read more about them in the official papers.

    - [Link to MEMENTO-3D-PET paper](https://ieeexplore.ieee.org/abstract/document/9660102)
    - [Link to MEMENTO-Multimodal paper](https://www.nature.com/articles/s41598-024-56001-9)

    Feel free to reach out for any inquiries or to get more information.
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


def check_dimensions(arr):
    return arr.shape == (128, 128, 50)


def main():
    st.sidebar.title("**MEMENTO**")
    selected_page = st.sidebar.radio(
        "Navigation", ["Try the tool!", "About", "Code"], label_visibility="hidden"
    )

    if selected_page == "Try the tool!":
        # st.title("MEMENTO")
        st.title("Automated detection of Alzheimer's Disease through PET and MRI scans")
        st.write(
            "Select a model from the dropdown menu and upload the corresponding image(s)."
        )
        st.write("Don't have any images? Download sample files below.")
        download_sample_files()

        # Add model selection dropdown on the main page
        selected_model_name = st.selectbox(
            "Select Model", ["MEMENTO-3D-MRI", "MEMENTO-3D-PET", "MEMENTO-MULTIMODAL"]
        )

        # Select the appropriate model based on the user's choice
        selected_model = select_model(selected_model_name)

        uploaded_file_1 = st.file_uploader(
            "Upload 3D MRI Scan: allowed dimension is (128x128x128)", type=["npy"]
        )
        uploaded_file_2 = None
        img_array_2 = None

        if selected_model_name == "MEMENTO-MULTIMODAL":
            uploaded_file_2 = st.file_uploader(
                "Upload PET Scan: allowed dimension is (128x128x128)", type=["npy"]
            )

        if uploaded_file_1 is not None and (
            uploaded_file_2 is None or uploaded_file_2 is not None
        ):
            try:
                img_array_1 = np.load(uploaded_file_1)

                if uploaded_file_2 is not None:
                    img_array_2 = np.load(uploaded_file_2)

                if (
                    selected_model == memento_multimodal_model
                    and (uploaded_file_2 is None or img_array_2 is None)
                ) or not check_dimensions(img_array_1):
                    st.warning(
                        "Uploaded file dimensions or model compatibility are not valid. Please upload valid files."
                    )
                else:
                    st.success("File dimensions and model compatibility are valid!")

                    match selected_model_name:
                        case "MEMENTO-3D-MRI":
                            st.text("Displaying uploaded MRI (central frame):")
                            st.image(img_array_1[:, :, 25])
                        case "MEMENTO-3D-PET":
                            st.text("Displaying uploaded PET (central frame):")
                            st.image(img_array_1[:, :, 25])
                        case "MEMENTO-MULTIMODAL":
                            st.text("Displaying uploaded MRI (central frame):")
                            st.image(img_array_1[:, :, 25])
                            st.text("Displaying uploaded PET (central frame):")
                            st.image(img_array_2[:, :, 25])

                    prediction = predict(img_array_1, img_array_2, selected_model)

                    st.subheader("Prediction:")
                    st.write(
                        f"The predicted probability of Alzheimer's Disease is: {prediction}"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

    elif selected_page == "About":
        about_section()

    elif selected_page == "Code":
        github_section()


if __name__ == "__main__":
    main()
