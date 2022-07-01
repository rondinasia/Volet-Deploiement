
# imports

import tensorflow as tf 
import numpy as np
import os
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from flask import Flask, render_template, request

# Create flask app
app = Flask(__name__)

@app.route('/', methods =['GET'] )

#render out pre-built HTML file right on the index page
def acceuil():
	return render_template("index.html",pagefirstload=True)

@app.route('/', methods =['POST'] )



#a form to upload the image by the user & send it to the Flask server so that the model is applied 
def predict():

    imagefile = request.files["imagefile"]

    # change those lines to os method of saving an image 
    image_path = "static/ImagesPVSuspect/" + imagefile.filename
    imagefile.save(image_path)

    #image pre-processing
    image = load_img(image_path,grayscale=True, target_size=(100,100))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

    #load the model 
    model = tf.keras.models.load_model("model/model_6c_11aug_5r_100p_Smote.h5")
    output = model.predict(image)*100
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    
    if np.argmax(output)==0 :  
        valeur = False # Pas de defaut 
        class_probability = None
        label_class = None

        # A changer 
        image_path = "static/ImagesWithoutDefect/" + imagefile.filename
        imagefile.save(image_path)

    else :
        valeur = True # Le defaut existe  
        label_class = np.argmax(output)  # get the class index with the highest probability 
        class_probability = np.max(output) #how likely the defaut belongs to this class
        image_path = "static/ImagesWithDefect/Class" + str(label_class) +'/' +  imagefile.filename
        imagefile.save(image_path)
    
    
 
    return render_template("index.html",exist = valeur ,prediction = class_probability ,prediction2 = label_class, pagefirstload=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT',8000))
    app.run(debug=True,host='0.0.0.0',port=port )
    