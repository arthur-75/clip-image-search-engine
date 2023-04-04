"""This module contains functions for ImageNet Similarity Search."""

import ipoly
import numpy as np
import pinecone
import tensorflow as tf
import streamlit as st


def text_embedding(text, processor, model):
    """Computes the text embedding for a given text input.

    Args:
        text (str): The text input to be embedded.
        processor (transformers.PreTrainedTokenizer): The tokenizer to process the input text.
        model (transformers.PreTrainedModel): The pretrained model to compute the embeddings.

    Returns:
        tf.Tensor: The normalized text embedding tensor.
    """
    tokens = processor(text=text, padding=True, images=None, return_tensors="tf")
    text_emb = model.get_text_features(**tokens)
    norm_factor = np.linalg.norm(text_emb, axis=1)
    text_emb = tf.transpose(text_emb) / norm_factor
    text_emb = tf.transpose(text_emb)
    return text_emb


def image_embedding(image_batch, processor, model):
    """Computes image embeddings for a batch of images.

    Args:
        image_batch (List[Union[PIL.Image.Image, np.ndarray]]): A list of images to be embedded.
        processor (transformers.PreTrainedTokenizer): The tokenizer to process the input images.
        model (transformers.PreTrainedModel): The pretrained model to compute the embeddings.

    Returns:
        tf.Tensor: The normalized image embedding tensor.
    """
    images = processor(text=None, images=image_batch, return_tensors="tf")[
        "pixel_values"
    ]
    img_emb = model.get_image_features(images)
    norm_factor = np.linalg.norm(img_emb, axis=1)
    img_emb = tf.transpose(img_emb) / norm_factor
    img_emb = tf.transpose(img_emb)
    return img_emb


def connect_pinecone(processor, model, api_key, environment="us-central1-gcp"):
    """Connects to Pinecone and initializes the index with image embeddings if it doesn't exist.

    Args:
        processor (transformers.PreTrainedTokenizer): The tokenizer to process the input images.
        model (transformers.PreTrainedModel): The pretrained model to compute the embeddings.
        api_key (str): The API key for Pinecone.
        environment (str, optional): The Pinecone environment to connect to. Defaults to "us-central1-gcp".

    Returns:
        pinecone.Index: The Pinecone index containing the image embeddings.
    """
    pinecone.init(api_key=api_key, environment=environment)
    index_name = "imagenet-index"
    if not (index_name in pinecone.list_indexes()):
        dataset = ipoly.load("imagenet-sample/data", keep_3D=True, file_names=True)
        np_dataset = np.array(dataset)
        img_embd = image_embedding(np_dataset[:, 0].tolist(), processor, model)
        index_vectors = list(zip(np_dataset[:, 1].tolist(), img_embd.numpy().tolist()))
        pinecone.create_index(index_name, dimension=512)
        index = pinecone.Index(index_name)
        index.upsert(index_vectors)
    else:
        index = pinecone.Index(index_name)
    return index


def get_images(text_input, processor, model, index):
    """Finds the top 5 most similar images to the input text from the Pinecone index.

    Args:
        text_input (str): The input text to find related images.
        processor (transformers.PreTrainedTokenizer): The tokenizer to process the input text and images.
        model (transformers.PreTrainedModel): The pretrained model to compute the embeddings.
        index (pinecone.Index): The Pinecone index containing the image embeddings.

    Returns:
        List[Tuple[str, float]]: A list of tuples with image names and their similarity scores.
    """
    test_embd = text_embedding(text_input, processor, model).numpy().tolist()
    return index.query(test_embd, top_k=5, include_metadata=True)


if __name__ == "__main__":
    st.title("Image seeker")

    api_token = st.text_input("Enter your API token", type="password")
    text_input = st.text_input("Enter a small text")

    model, processor = ipoly.load_transformers("openai/clip-vit-base-patch32")

    if st.button("Validate"):
        if not api_token:
            st.error("Please enter a valid API token.")
        else:
            index = connect_pinecone(processor, model, api_token)
            images = get_images(text_input, processor, model, index)
            images = [name["id"] for name in images["matches"]]
            if images:
                for img in images:
                    st.image(img, caption="Image", use_column_width=True)
            else:
                st.error("No images were found.")
