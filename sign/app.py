from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import predict_signature

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'reference' not in request.files or 'verification' not in request.files:
            return redirect(request.url)
        ref_file = request.files['reference']
        ver_file = request.files['verification']
        if ref_file.filename == '' or ver_file.filename == '':
            return redirect(request.url)
        if ref_file and ver_file:
            ref_filename = secure_filename(ref_file.filename)
            ver_filename = secure_filename(ver_file.filename)
            ref_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
            ver_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ver_filename)
            ref_file.save(ref_filepath)
            ver_file.save(ver_filepath)
            
            # Make prediction for the reference and verification signatures
            ref_class, ref_prob = predict_signature(ref_filepath)
            ver_class, ver_prob = predict_signature(ver_filepath)
            
            result = 'Genuine' if ref_class == ver_class and ver_class == 'Genuine' else 'Forged'
            match_prob = min(ref_prob, ver_prob)

            return render_template('index.html', ref_filename=ref_filename, ver_filename=ver_filename, 
                                   result=result, match_prob=match_prob)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
