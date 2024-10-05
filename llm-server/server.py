import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.distiled_gpt2 import DistiledGPT2
# Create a Flask app instance
app = Flask(__name__)
CORS(app)

cached_result = ''

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
    model.load('distilgpt2-finetuned-epoch5')
    cached_result = result = model.predict(input_value)
    model.unload()
    return jsonify({"result": result})


# Run the app
if __name__ == '__main__':
    device = torch.device('cuda')
    model = DistiledGPT2(device)
    app.run(debug=True, port=5005)