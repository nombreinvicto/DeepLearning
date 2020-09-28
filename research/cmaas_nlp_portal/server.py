# library imports
import os, json, pickle, numpy as np
from os import path, walk
from flask_cors import CORS
from flask import Flask, request, redirect, url_for, flash
from flask import render_template
from inference import part_types, cad_dir
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename

# global variable initialisations
cwd = os.getcwd()
app = Flask(__name__)
app.env = "development"
flask_app_PORT = 3000
pad_max_length = 4
part_categories = ['bearing', 'bolt', 'collet', 'spring', 'sprocket']
save_path = r"C:\Users\mhasa\AppData\Roaming\Autodesk\Autodesk Fusion 360\MyScripts\PureFlask_3JS_server\flask_app\nlp_model"


# my declarations
# load the keras trained model
with open(f'{save_path}//tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# load the keras model
loaded_model = load_model(f"{save_path}//nlp_model.h5")



# jina2 doesnt have these functions. this is the way
# we make sure jina2 templating language what sth means
app.jinja_env.globals['zip'] = zip
app.jinja_env.globals['enumerate'] = enumerate
CORS(app)

# watch for extra directories
extra_dirs = ['./static', './templates', ]
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

flaskKwargs = {'debug': True, 'port': flask_app_PORT,
               'extra_files': extra_files}


# routing
@app.route('/')
def homepage():
    # init all the variables according to context
    query_string_dict = request.args
    supplied_part = query_string_dict.get('part')
    default_title = "DIME Labs CAD Search NLP Engine"
    image_src_tags = []
    list_of_images = []

    # see if this request has been made to query a part or not
    if supplied_part:
        # construct the path to the images
        html_image_path = f"../static/cad_repo/{supplied_part}/images"
        os_image_path = f"{cwd}/static/cad_repo/{supplied_part}/images"
        list_of_images = os.listdir(os_image_path)

        # now construct the list of relative src for html img tag
        for image in list_of_images:
            image_src_tags.append(f"{html_image_path}/{image}")

    # construct the context_dict
    context = {'title': default_title,
               'part': supplied_part,
               'image_src_tags': image_src_tags,
               'image_list': list_of_images}

    return render_template('homepage.html', **context)


@app.route('/parse_text', methods=['GET', 'POST'])
def parse_form_text():
    supplied_text = request.form['text']

    if supplied_text:
        query_text_for_nlp = np.asarray([supplied_text])
        text_sequence = loaded_tokenizer.texts_to_sequences(query_text_for_nlp)
        padded_sequence = pad_sequences(text_sequence, maxlen=pad_max_length)
        nlp_pred = loaded_model.predict(padded_sequence)
        index = np.argmax(nlp_pred)
        queried_part = part_categories[int(index)]

        query_string = {'part': queried_part}
        return redirect(url_for("homepage", **query_string))

    else:
        return "No Text Supplied"


@app.route('/open_cad')
def open_cad_in_f360():
    # init all the variables according to context
    query_string_dict = request.args
    supplied_part = query_string_dict.get('part')
    supplied_filename = query_string_dict.get('id')

    print(supplied_part)
    print(supplied_filename)

    return supplied_part, supplied_filename


# server spawn
if __name__ == '__main__':
    app.run(**flaskKwargs)
