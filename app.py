from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the pre-trained model
model = load_model('FV.h5')

# Define class labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the file upload
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            # Preprocess the image and make predictions
            input_data = preprocess_image(file_path)
            predictions = model.predict(input_data)

            # Get the predicted class
            predicted_class = np.argmax(predictions)
            predicted_label = labels[predicted_class]

            # Render the result on the HTML page
            return render_template('result.html', image_path=file_path, predicted_label=predicted_label)

    return render_template('index.html')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == '__main__':
    app.run(debug=True)
