from IPython.display import HTML, display
from tokenizers import Tokenizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly import express as px
import pandas as pd
from ollama import Client


# Generates a unique HSL color based on the index of the token
def _number_to_color(number: int) -> str:
    golden_ratio_conjugate = 0.618033988749895
    a = 1664525
    c = 1013904223
    m = 2**32

    pseudorandom = (a * number + c) % m
    hue = ((pseudorandom * golden_ratio_conjugate) % 1) * 360
    saturation = 60 + (pseudorandom % 21)
    lightness = 70 + (pseudorandom % 21)

    return f"hsl({hue}, {saturation}%, {lightness}%)"


# Generates a dictionary mapping unique texts/tokens with their corresponding color
def _get_token_colors(tokens) -> dict:
    unique_tokens = list(set(tokens))
    token_colors = {
        token: _number_to_color(index) for index, token in enumerate(unique_tokens)
    }
    return token_colors


def visualize_tokens(text: str, tokenizer_name: str):
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    tokenized = tokenizer.encode(text)
    tokens = [token.replace("Ġ", "") for token in tokenized.tokens]
    color_map = _get_token_colors(tokens)

    html_content = '<div style="background-color: white; padding: 20px;">'
    # Add tokenizer title
    html_content += f'<h3 style="color: #3366cc; font-family: Arial, sans-serif;">{tokenizer_name}</h3>'

    colored_text = ""
    for token in tokens:
        color = color_map[token]
        colored_text += f'<span style="color: black; background-color: {color}; padding: 2px;">{token}</span> '

    html_content += f"{colored_text}</div>"
    display(HTML(html_content))


def bulk_embed(model_name: str, inputs: list[str], client: Client) -> np.ndarray:
    """
    Generate embeddings for the given inputs using the specified model.
    """
    result = client.embed(
        model=model_name,
        input=inputs,
    )
    embeddings = result.embeddings

    # Convert to numpy array for processing
    embeddings = np.array(embeddings)
    return embeddings


def plot_similarity_heatmap(
    similarities: np.ndarray, texts: list[str], limit: int = 15
):
    wrapped_texts = [
        text[: limit - 3] + "..." if len(text) > limit else text for text in texts
    ]
    fig = px.imshow(
        similarities,
        labels=dict(x="Texts", y="Texts", color="Similarity"),
        x=wrapped_texts,
        y=wrapped_texts,
    )
    fig.update_xaxes(side="top", tickangle=45)
    fig.update_yaxes(tickangle=-45)
    fig.show()


def plot_embeddings(embeddings: np.ndarray, texts: list[str]) -> None:
    """
    Plot a 3D PCA visualization of the embeddings.
    """
    # Standardize the data (mean removal and scaling)
    scaler = StandardScaler()
    standardized_embeddings = scaler.fit_transform(embeddings)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3, random_state=42)
    pca_embeddings = pca.fit_transform(standardized_embeddings)

    # Prepare DataFrame with hover text and colors
    df = pd.DataFrame(
        {
            "x": pca_embeddings[:, 0],
            "y": pca_embeddings[:, 1],
            "z": pca_embeddings[:, 2],  # Third dimension from PCA
            "text": texts,
        }
    )

    # Create the scatter plot with hover labels
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        title=f"3D PCA Visualization",
        color="text",
        hover_data=["text"],
    )

    def axes_style3d(
        bgcolor="rgb(20, 20, 20)", gridcolor="rgb(150, 150, 150)", zeroline=False
    ):
        return dict(
            showbackground=True,
            backgroundcolor=bgcolor,
            gridcolor=gridcolor,
            zeroline=False,
        )

    my_axes = axes_style3d(
        bgcolor="rgba(0, 0, 0, 0)",
        gridcolor="rgb(100, 100, 100)",
    )
    # Update layout to show axes and remove grids
    fig.update_layout(
        template="none",
        margin=dict(t=2, r=2, b=2, l=2),
        scene=dict(xaxis=my_axes, yaxis=my_axes, zaxis=my_axes),
    )

    fig.show()


if __name__ == "__main__":
    tokenizer = Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    text = "Hello World!"
    token_colors = visualize_tokens(text, tokenizer)
