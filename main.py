from fastapi import FastAPI, File, UploadFile, HTTPException
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

    df_embeddings = pd.read_csv(embedding_path, index_col=0)
    embeddings_array = df_embeddings.values

    styles_df = pd.read_csv(styles_path, nrows=6000, on_bad_lines="skip")
    styles_df["image"] = styles_df["id"].apply(lambda x: f"{x}.jpg")

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
    return styles_df["image"].iloc[[i[0] for i in sorted_indices]].tolist()


@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    print("here")
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


# For debugging purposes
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
