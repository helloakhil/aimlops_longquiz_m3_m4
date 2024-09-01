# Set PATH
from http.client import HTTPException
import json
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
sys.path.append(str(file.parents[2]))

# 1. Library imports
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi import FastAPI

from iris_classifier.predict import make_prediction
from iris_classifier.processing.validation import IrisSpeciesInput

# 2. Create app and model objects
app = FastAPI()

@app.get('/')
def home():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

@app.post('/predict')
async def predict_species(iris: IrisSpeciesInput):
    data = iris.dict()
    results = make_prediction(input_data=data)

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results


# 4. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)