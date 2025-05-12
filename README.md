# Fashion Recommendation Backend

This is the backend API for the Fashion Recommendation System. It is built with Python and provides endpoints for generating and serving fashion recommendations.

---

## Features

- REST API for fashion recommendations
- Embedding-based item similarity
- CSV-based data storage

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

---

## Setup Instructions

1. **Clone the repository and navigate to the backend folder:**
   ```sh
   cd backend
   ```
2. **(Optional) Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Python Version for TensorFlow

TensorFlow 2.15+ does not support Python 3.12 as of May 2025. To use TensorFlow, you must use Python 3.11 or lower.

### How to Downgrade to Python 3.11 on Windows

1. Download Python 3.11 from the official website:
   https://www.python.org/downloads/release/python-3110/
2. Install Python 3.11. During installation, check "Add Python to PATH".
3. (Optional but recommended) Create a new virtual environment using Python 3.11:
   ```pwsh
   py -3.11 -m venv venv
   .\venv\Scripts\activate
   ```
4. Reinstall dependencies:
   ```pwsh
   pip install -r requirements.txt
   ```

After these steps, TensorFlow should install without issues.

---

## Running the Backend

1. **Start the API server:**

   ```sh
   python main.py
   ```

   The server will start on the default port (e.g., 8000).

2. **API Endpoints:**
   - Main recommendation endpoint: `/recommend` (see `main.py` for details)

---

## Deployment

- The backend can be deployed to platforms like Heroku or Vercel (see `Procfile` and `vercel.json`).

---

## Project Structure

```
backend/
  main.py              # Main API server
  requirements.txt     # Python dependencies
  df_embeddings.csv    # Embedding data
  images.csv           # Image metadata
  styles.csv           # Style metadata
  Procfile             # For deployment
  vercel.json          # For Vercel deployment
```

---

## License

MIT License
