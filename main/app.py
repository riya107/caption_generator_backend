import os
import uuid
from flask import Flask, request
from flask_cors import CORS
import joblib
import keras
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from keras.models import Model
import numpy as np

app = Flask(__name__)

CORS(app)

tf.config.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(True)

model = keras.models.load_model("caption_generator.h5")
word2ind = joblib.load('word2ind.pkl')
ind2word = joblib.load('ind2word.pkl') 
encodings = joblib.load('encodings.pkl') 

pre_trained_model = InceptionV3(weights = 'imagenet')
encoder = Model(inputs=pre_trained_model.input, outputs=pre_trained_model.layers[-2].output)

largest_caption_size = 34

def preprocessing_for_inceptionv3(path):
    img = load_img(os.path.join('/content/Images/',path), target_size = (299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img
  
def encode(image):
    image = preprocessing_for_inceptionv3(image)
    image = encoder.predict(image)
    image = image.reshape((image.shape[1],))
    return image

def generate_caption(encoding):
	sequence = '<start>'
	for i in range(largest_caption_size):
		seq = [word2ind[word] for word in start.split() if word in word2ind]
		seq = pad_sequences([seq], maxlen = largest_caption_size)[0]
		seq = seq.reshape(1,len(seq))
		pred = model.predict([encoding, seq])
		pred = np.argmax(pred)
		word = ind2word[pred]
		start += ' ' + word
		if word == '<end>':
			break
	sequence = sequence.split(' ')
	sequence = sequence[1:-1]
	sequence = ' '.join(sequence)
	return sequence

@app.route("/generateCaption",methods=["POST"])
def generateCaption():
	file = request.files['file']
	file_extension = file.filename.split('.')[1]
	file.filename = str(uuid.uuid4())+'.'+file_extension
	file.save(os.path.join('uploads', file.filename))
	filename = os.path.join('uploads', file.filename)
	encoding = encode(filename)
	os.remove(filename)
	encoding = encoding.reshape(1,2048)
	return {"caption":generate_caption(encoding)}


if __name__=='__main__':
    app.run(host='127.0.0.1',port=8000,debug=True)