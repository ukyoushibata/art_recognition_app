import os
import gdown
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["Albrecht_Durer","Edgar_Degas","Pablo_Picasso","Paul_Gauguin","Pierre-Auguste_Renoir","Vincent_van_Gogh"]
image_size = 224

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


MODEL_PATH = "model/model.46-0.71.model.keras"
# DRIVE_FILE_ID = "1ALcSlF-ct1hEUTsWlo4oQKdKRAz5lCz-"  

# # モデルをローカルにダウンロード（なければ）
# if not os.path.exists(MODEL_PATH):
#     print("Downloading model...")
#     os.makedirs("model", exist_ok=True)
#     gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# モデルを読み込み
model = load_model(MODEL_PATH)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(image_size,image_size,3))
            img = image.img_to_array(img)
            img /= 255.0
            img = np.expand_dims(img, axis=0)
            #data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(img)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)