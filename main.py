from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D  # type: ignore
from sklearn.metrics.pairwise import linear_kernel
import os
import tempfile
import requests

# Initialize FastAPI app first
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Then load data and model
try:
    # Load embeddings and styles data
    df_embeddings = pd.read_csv("./df_embeddings.csv", index_col=0)
    embeddings_array = df_embeddings.values

    styles_df = pd.read_csv("./styles.csv", nrows=6000, on_bad_lines="skip")
    styles_df["image"] = styles_df["id"].apply(lambda x: f"{x}.jpg")

    images_df = pd.read_csv("./images.csv", nrows=6000, on_bad_lines="skip")

    # Create safer mapping with fallback
    filename_to_link = dict(zip(images_df["filename"], images_df["link"]))
    styles_df["link"] = styles_df["image"].map(lambda x: filename_to_link.get(x, ""))

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
    print(recommended_ids)
    # Get links with fallback
    return [
        styles_df[styles_df["image"] == recommended_id].to_dict(orient="records")[0]
        for recommended_id in recommended_ids
    ]


@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            image.load_img(temp_path, target_size=(100, 100))
        except Exception as e:
            raise HTTPException(400, "Invalid image file") from e

        return {"recommendations": generate_recommendations(temp_path)}
    finally:
        os.remove(temp_path)


@app.post("/recommendations/url/")
async def get_recommendations_from_url(url: str = Body(..., embed=True)):
    """New endpoint for URL-based recommendations"""
    temp_path = None
    try:
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "Invalid URL scheme")

        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Verify image content
        if "image/" not in response.headers.get("Content-Type", ""):
            raise HTTPException(400, "URL does not point to an image")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        # Validate image
        try:
            image.load_img(temp_path, target_size=(100, 100))
        except Exception as e:
            raise HTTPException(400, "Invalid image file") from e

        return {"recommendations": generate_recommendations(temp_path)}
    except requests.RequestException as e:
        raise HTTPException(400, f"Failed to download image: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/items")
async def get_items(page: int = 1, page_size: int = 100):
    """Get paginated items with safety checks"""
    if page < 1 or page_size < 1:
        raise HTTPException(400, "Page and page_size must be ≥ 1")

    total_items = len(styles_df)
    total_pages = (total_items + page_size - 1) // page_size

    start = (page - 1) * page_size
    end = start + page_size

    # Clamp values to valid ranges
    start = max(0, start)
    end = min(end, total_items)

    if start >= total_items:
        return {
            "items": [],
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_items": total_items,
        }

    # Clean NaN values before serialization
    items_data = (
        styles_df.iloc[start:end].fillna("").to_dict(orient="records")
    )  # <-- Fix here

    return {
        "items": items_data,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_items": total_items,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


@app.get("/item")
async def get_item(id: int):
    """Get item by ID"""
    # get item data by matching ID with style "id" column
    item_data = styles_df[styles_df["id"] == id].to_dict(orient="records")[0]

    if (item_data["image"] is None) or (item_data["link"] is None):
        raise HTTPException(404, "Item not found")

    return item_data


@app.get("/")
async def root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
