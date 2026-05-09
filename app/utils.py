from PIL import Image
import numpy as np
from io import BytesIO

def read_image(file):

    image = Image.open(BytesIO(file)).convert("RGB")

    image = image.resize((32,32))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    return image