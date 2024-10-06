import os
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.gpt import *
# Create a Flask app instance
app = Flask(__name__)
CORS(app)

cached_result = ''

script_dir = os.path.dirname(os.path.realpath(__file__))

# Define a route that returns JSON data
@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    input_value = data.get('input', None)
    if input_value is None or len(input_value) == 0:
        return jsonify({"error": "An error occurred while processing the request."}), 400
    
    global cached_result
    if cached_result.startswith(input_value):
        return jsonify({"result": cached_result})
    
    cached_result = result = model.predict(input_value)
    return jsonify({"result": result})


# Run the app
if __name__ == '__main__':
    model = GPTModel((os.path.join(script_dir, '../distilgpt2-finetuned')), device=torch.device('cuda'))
    app.run(debug=False, port=5005)