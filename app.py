from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg19 import VGG19

app = Flask(__name__)

# Initializing a vgg model. In this project i am using vgg 19 with weights from imagenet

vgg = VGG19(weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False         # making all the layers non trainable as we have already imported model with imagenet weights

# this fxn converts the vgg model as per our output classes. As we want to have only 1 class at the input we need to modify the vgg model
def give_model(optimizer='adam', vgg=vgg):  
    x = vgg.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(8000, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=out)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = give_model()
# We have already developed a model using the dataset and saved the best val_accuracy model and loaded here.

model.load_weights(
    'D:\Data Analytics for Business\Keras learning\cat_dog_pred\model_to_load\cat-dog.hdf5')

# give_reslut will do the preprocessing that need to be done on any image the provides us with and then returns the predictions

def give_result(filename):
    path = os.path.join('D:\Data Analytics for Business\Keras learning\cat_dog_pred\static', filename)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (224, 224))
    im = im.reshape(-1, 224, 224, 3)
    return model.predict(im)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/image', methods=['GET', 'POST'])
def show_image():
    if request.method == 'POST':
        im = request.files['file']
        filename = im.filename
        im.save(os.path.join('D:\Data Analytics for Business\Keras learning\cat_dog_pred\static', filename))
        res = give_result(filename)     # using give_result converting the input image to our desired size and predicting on the model
        print(res)
    return render_template('index.html', im=filename, res=res)


@app.route('/display/<filename>')
def display_image(filename):

    return redirect(url_for('static', filename='/'+filename), code=301)


if __name__ == '__main__':
    app.run(debug=True, port=7000)
