"""This module contains functions for ImageNet Similarity Search.

The module contains the following functions:
- `search_index`: Searches for text input and returns a list of images.
- `connect_pinecone`: Connects to Pinecone and returns an index.
- `get_model`: Gets the model and tokenizer for a specified name.
- `get_dataset`: Gets the ImageNet Sample dataset.
- `__main__`: The main function of the ImageNet Similarity Search. It displays a text input for the
  user to enter a text description of an image, and displays the top matching images in Streamlit.
"""
import fiftyone as fo
import numpy as np
import pinecone
import streamlit as st
import tensorflow as tf
from fiftyone import ViewField as FV
from PIL import Image
from transformers import AutoTokenizer
from transformers import TFAutoModel


def search_index(text_input, tokenizer, model, index, dataset, n_results=5):
    """Searches for text input and returns a list of images.

    Args:
        text_input (str): The input text to search for.
        tokenizer (function): A function to tokenize the input text.
        model (object): An object representing the model used to get text features.
        index (object): An object representing the index to search for the input text.
        dataset (object): An object representing the dataset containing the images.
        n_results (int, optional): The number of results to return. Defaults to 5.

    Returns:
        list: A list of dictionaries, where each dictionary represents an image and has two keys:
              "id" and "data". The "id" key contains the id of the image and the "data" key contains
              the data of the image in the form of a numpy array.
    """
    text_input = tokenizer(text_input, return_tensors="tf", padding=True)
    text_features = model.get_text_features(text_input)[0].numpy()

    query_embedding = np.squeeze(text_features)
    results = index.query(query_embedding, k=n_results)

    if results is None:
        return []

    result_ids = [r.id for r in results]
    matching_images = dataset.filter(FV("metadata.id").is_in(result_ids)).take(
        n_results,
    )

    image_list = []
    for image in matching_images:
        img = Image.fromarray(image["pixels"])
        img = img.resize((224, 224))
        img_data = np.array(img)
        image_list.append({"id": image["metadata"]["id"], "data": img_data})

    return image_list


def connect_pinecone(api_key, environment="us-central1-gcp"):
    """Connects to Pinecone and returns an index.

    Args:
        api_key (str): The API key used to connect to Pinecone.
        environment (str, optional): The environment to connect to. Defaults to "us-central1-gcp".

    Returns:
        object: An object representing the Pinecone index.
    """
    pinecone.init(api_key=api_key, environment=environment)
    index_name = "imagenet-index"
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    pinecone.create_index(index_name, dimension=512)
    return pinecone.Index(index_name)


def get_model(model_name="openai/clip-vit-base-patch32"):
    """Gets the model and tokenizer for a specified name.

    Args:
        model_name (str, optional): The name of the model to get. Defaults to "openai/clip-vit-base-patch32".

    Returns:
        tuple: A tuple containing the model and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    return model, tokenizer


def get_dataset():
    """Gets the ImageNet Sample dataset.

    Returns:
        object: An object representing the ImageNet Sample dataset.
    """
    dataset = fo.zoo.load_zoo_dataset("imagenet-sample")

    session = fo.launch_app(dataset, address="0.0.0.0")
    return dataset


if __name__ == "__main__":
    """The main function of the ImageNet Similarity Search.

    This function displays a text input for the user to enter a text
    description of an image, and displays the top matching images in
    Streamlit.
    """
    st.title("ImageNet Similarity Search")

    text_input = st.text_input("Enter a text description of an image")
    if text_input:
        results = search_index(text_input)
        if results:
            st.write(f"Top {len(results)} matching images:")
            for result in results:
                st.image(result["data"], caption=result["id"], width=224)
        else:
            st.write("No matching images found.")

