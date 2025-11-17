import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

def run_classification(filename: str):
    '''
    Classify the scribble into one of the imagenet classes using the ResNet50 model

    Params:
        filename (str): name of the scribble image with filetype extension
    
    Returns:
        y_pred: predicted class of the image
    '''

    image = load_img(f'{filename}', target_size=(224, 224))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)

    model = ResNet50(weights='imagenet')

    y_pred = model.predict(arr)
    
    y_pred = decode_predictions(y_pred, top=1)[0][0][1]
    y_pred = "".join([char if i != 0 else char.upper() for i, char in enumerate(y_pred)])
    y_pred = y_pred.replace("_", " ")

    return y_pred

