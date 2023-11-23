from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
from model.script import DetectInpaint

app = Flask(__name__)

@app.route('/', methods=['POST'])
def convert_image():
    try:
        base64_image = request.json['image']
        task = request.json['task']
        
        image_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_data))
        
        detect_inpaint = DetectInpaint(
            image=img,
            use_cuda_if_available=False,
            task=task
        )
        img = detect_inpaint.run()
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = {
            'image': img_base64
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'message': 'An error occurred during image conversion', 'error': str(e)}), 400

if __name__ == '__main__':
    app.run()