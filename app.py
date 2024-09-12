from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import pandas as pd
from urllib.parse import urlparse

app = FastAPI()

# Specify the directory for templates
templates = Jinja2Templates(directory="templates")

# Load the model from the pickle file
with open("cyber attack.pkl", "rb") as f:
    model = pickle.load(f)

# Function to extract and combine features from a URL
def extract_features_from_url(url: str) -> pd.DataFrame:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    
    # Extract individual features (this should match with the features used in your model)
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)
    
    # Example feature extraction; adjust as per your model's requirements
    features = {
        'Domain Length': domain_length,
        'Path Length': path_length,
        'Query Length': query_length,
        # Add more features as needed
    }
    
    # Combine features into a single feature
    combined_feature = sum(features.values())
    
    # Create DataFrame with the combined feature
    return pd.DataFrame([[combined_feature]], columns=['Combined_Feature'])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(url: str = Form(...)):
    # Extract features from the URL
    data = extract_features_from_url(url)
    
    # Make prediction using the loaded model
    prediction = model.predict(data)
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": prediction[0]})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
