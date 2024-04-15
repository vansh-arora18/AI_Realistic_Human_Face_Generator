from flask import Flask, render_template, send_file
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
import io
import os
import random

app = Flask(__name__)

# Load your trained machine learning model
generator = load_model('generator1.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index1.html')

# Define a route to serve the generated image
@app.route('/generate_image')
def generate_image():
    # generate new image
    noise = tf.random.normal([1, 100])
    fig = plt.figure(figsize=(3, 3))
    # generate the image from noise
    g_img = generator(noise)
    # denormalize the image
    g_img = (g_img * 127.5) + 127.5
    g_img.numpy()
    img = array_to_img(g_img[0])
    plt.imshow(img)
    plt.axis('off')
    # Save the generated image to a byte buffer
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='jpeg')
    img_bytes.seek(0)
    # Close the figure to avoid memory leaks
    plt.close(fig)
    # Serve the generated image
    BASE_DIR = 'Humans/'
    image_paths = [os.path.join(BASE_DIR, image_name) for image_name in os.listdir(BASE_DIR)]

    image_paths_new = random.choice(image_paths)

    return send_file(image_paths_new, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
