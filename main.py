from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import linear_kernel
import os
import tempfile

# Initialize FastAPI app first
app = FastAPI()

# Then load data and model
try:
    # Load embeddings and styles data
    embedding_path = "./df_embeddings.csv"
    styles_path = "./styles.csv"
    images_path = "./images.csv"

    df_embeddings = pd.read_csv(embedding_path, index_col=0)
    embeddings_array = df_embeddings.values

    styles_df = pd.read_csv(styles_path, nrows=6000, on_bad_lines="skip")
    styles_df["image"] = styles_df["id"].apply(lambda x: f"{x}.jpg")

    images_df = pd.read_csv(images_path, nrows=6000, on_bad_lines="skip")

    # add the link column from images_df to styles_df using the filename column
    styles_df["link"] = styles_df["image"].map(
        lambda x: images_df[images_df["filename"] == x]["link"].values[0]
    )

    # Load and configure VGG16 model
    vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=(100, 100, 3))
    vgg16.trainable = False
    model = Sequential([vgg16, GlobalMaxPooling2D()])

except Exception as e:
    raise RuntimeError(f"Initialization failed: {str(e)}")


def generate_recommendations(image_path: str) -> list:
    """Generate fashion recommendations from an image path."""
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    sample_embedding = model.predict(img_array)
    similarity = linear_kernel(sample_embedding, embeddings_array)

    # Normalization
    x_min, x_max = similarity.min(axis=1), similarity.max(axis=1)
    norm_similarity = (similarity - x_min) / (x_max - x_min)[:, np.newaxis]

    # Get top recommendations
    sorted_indices = sorted(
        enumerate(norm_similarity[0]), key=lambda x: x[1], reverse=True
    )[:5]
    recommended_ids = styles_df["image"].iloc[[i[0] for i in sorted_indices]].tolist()

    # Create a mapping from filename to link
    filename_to_link = dict(zip(images_df["filename"], images_df["link"]))
    # Get the corresponding links in order
    recommended_links = [filename_to_link[filename] for filename in recommended_ids]

    return recommended_links


@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Validate image
        try:
            image.load_img(temp_path, target_size=(100, 100))
        except Exception as e:
            raise HTTPException(400, "Invalid image file") from e

        return {"recommendations": generate_recommendations(temp_path)}

    finally:
        os.remove(temp_path)


@app.get("/items")
async def get_items(page: int = 1, page_size: int = 10):
    """Get a list of items."""
    return {
        "items": styles_df.iloc[page * page_size : (page + 1) * page_size].to_dict(
            orient="records"
        ),
        "total": styles_df.shape[0],
        "page": page,
        "page_size": page_size,
        "pages": styles_df.shape[0] // page_size + 1,
        "has_next": page < styles_df.shape[0] // page_size,
        "has_prev": page > 1,
        "next": page + 1 if page < styles_df.shape[0] // page_size else None,
        "prev": page - 1 if page > 1 else None,
        "first": 1,
        "last": styles_df.shape[0] // page_size + 1,
    }


# For debugging purposes
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
