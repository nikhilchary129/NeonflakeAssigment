

---

# Image Captioning and Emotion Classification Script

This Python script performs image captioning and emotion classification based on the generated caption. It uses the Salesforce BLIP model for generating captions and a BERT-based model for emotion classification.

# Colab Notebooks

You can run and interact with the following notebooks directly in Google Colab:



[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MK8Cl6DBbRiEVCn_5dioM_8xRg0kpWYT)

Click the button above to open the notebook in Google Colab.


## Prerequisites

Ensure you have the following libraries installed:

- `Pillow` for image processing
- `requests` for downloading images
- `transformers` for loading pre-trained models and tokenizers
- `torch` for tensor operations
- `IPython` for displaying images

You can install these libraries using pip:

```bash
pip install pillow requests transformers torch ipython
```

## Code Breakdown

### Imports

```python
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
from IPython.display import display
```

- `Image` from PIL is used to handle image operations.
- `requests` is used to fetch images from a URL.
- `transformers` provides the processor and models for image captioning and emotion classification.
- `torch` is used for tensor operations and inference.
- `display` from `IPython` is used to show images in Jupyter Notebooks or IPython environments.

### Load Models and Processors

```python
# Load the processor and model for image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pre-trained emotion classification model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model_emotion = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
```

- `processor` and `model` are loaded for generating captions from images.
- `tokenizer` and `model_emotion` are loaded for classifying the emotion in the caption.

### Functions

#### `get_image_caption(image)`

```python
def get_image_caption(image):
    # Prepare the input for image captioning
    text = "A picture of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate the image caption with max_new_tokens to control the length
    outputs = model.generate(**inputs, max_new_tokens=50)  # Adjust 50 to the desired length

    # Decode the output tokens into a readable string
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
```

- Prepares the image and text for caption generation.
- Uses the model to generate a caption.
- Decodes and returns the caption as a string.

#### `get_emotion(caption)`

```python
def get_emotion(caption):
    # Tokenize the generated caption
    inputs_emotion = tokenizer(caption, return_tensors="pt")

    # Get the model predictions for emotion classification
    with torch.no_grad():
        outputs_emotion = model_emotion(**inputs_emotion)

    # Get the predicted emotion class
    logits = outputs_emotion.logits
    predicted_class_id = logits.argmax(-1).item()
    predicted_emotion = model_emotion.config.id2label[predicted_class_id]
    return predicted_emotion
```

- Tokenizes the caption.
- Uses the emotion classification model to predict the emotion.
- Returns the predicted emotion as a string.

### Main Function

```python
def main():
    # URL of the image
    url = input("Enter the URL of the image: ")

    try:
        # Load the image
        image = Image.open(requests.get(url, stream=True).raw)
        
        # Display the image 
        display(image)
        
        # Get and display the caption
        caption = get_image_caption(image)
        print(f"Generated Caption: {caption}")

        # Get and display the emotion
        emotion = get_emotion(caption)
        print(f"Predicted Emotion: {emotion}")

    except Exception as e:
        print(f"Error: {e}")
```

- Prompts the user for an image URL.
- Downloads and opens the image.
- Displays the image.
- Generates a caption for the image.
- Classifies the emotion in the caption.
- Prints the generated caption and predicted emotion.

### Entry Point

```python
if __name__ == "__main__":
    main()
```

- Ensures that the `main()` function runs when the script is executed directly.

## Usage

Run the script in a Python environment. You will be prompted to enter the URL of an image. The script will display the image, generate a caption, and classify the emotion of the caption.

---

