from fastapi import FastAPI, Depends, UploadFile, File
from pydantic import BaseModel # This has nothing to do with machine learning models, 
# parent class for everything that is strictly typed in pydantic
from torchvision import transforms
from torchvision.models import ResNet
from PIL import Image
import io
import torch
import torch.nn.functional as F

from model import load_model, load_transforms, CATEGORIES

# The result from the api is of a pydantic class
# strictly typed to return 
# a string for the category, here predicted label 
# a float for confidence, here the probability of the label
class Result(BaseModel):
    category: str
    confidence: float


# This creates an instance for the endpoint
app = FastAPI()


# This is going to happen on every single predict.
@app.post("/predict", response_model=Result)
async def predict(
        input_image: UploadFile = File(...),  # Placeholder for the file.
        model: ResNet = Depends(load_model),  # Every thread of the execution will have its own instance of the model 
                                              # at least 45 MB for each user. Flask has a single model and it takes turns in replying.
                                              # There are strategies for load balancing. You serve asynchronously until the load is higher
                                              #  then you start serving in a queue. 
                                              # with async, it won't start until load_model is executed, because of the "Depends" keyword
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    # Read the uploaded image
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert RGBA image to RGB image, to get rid of the alpha channel on dim=4
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply the transformations
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        # TODO: set up a breakpoint to understand outputs[0] and dim=0
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Map the predicted class index to the category
    # .item() takes only the value in the torch.tensor
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())