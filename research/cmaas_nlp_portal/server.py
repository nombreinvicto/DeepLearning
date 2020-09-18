# library imports
import os, json
from os import path, walk
from flask_cors import CORS
from flask import Flask, request, redirect, url_for, flash
from flask import render_template
from inference import part_types, cad_dir
from werkzeug.utils import secure_filename

# global variable initialisations
cwd = os.getcwd()
app = Flask(__name__)
app.env = "development"
flask_app_PORT = 3000
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
               'image_src_tags': image_src_tags}

    return render_template('homepage.html', **context)


@app.route('/parse_text', methods=['GET', 'POST'])
def parse_form_text():
    supplied_text = request.form['text']

    if supplied_text:
        if supplied_text in part_types:
            # if supplied text falls in part types, then render pictures
            query_string = {'part': supplied_text}
            return redirect(url_for("homepage", **query_string))




        else:
            return "Part type doesnt exist"
    else:
        return "No Text Supplied"


# server spawn
if __name__ == '__main__':
    app.run(**flaskKwargs)
