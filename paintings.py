import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

classes = ["Albrecht_Durer","Edgar_Degas","Pablo_Picasso","Paul_Gauguin","Pierre-Auguste_Renoir","Vincent_van_Gogh"]
image_size = 224

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

MODEL_PATH = "model/model_quant.tflite"

# TFLiteモデル読み込み
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(image_size, image_size))
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


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
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像を前処理
            input_data = preprocess_image(filepath)

            # TFLiteモデルに入力セット
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 推論実行
            interpreter.invoke()

            # 出力取得
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            predicted = np.argmax(output_data)
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)