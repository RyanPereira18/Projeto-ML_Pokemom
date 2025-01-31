import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Carregar o modelo e os nomes
try:
    model = pickle.load(open("model.pkl", "rb"))
    names = pickle.load(open("names.pkl", "rb"))
except Exception as e:
    print(f"Erro ao carregar o modelo ou nomes: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []
        for x in request.form.values():
            try:
                features.append(float(x))
            except ValueError:
                return render_template("index.html", prediction_text="Erro: Por favor, insira apenas números.")

        final_features = [np.array(features)]
        pred = model.predict(final_features)

        if len(pred) == 0:
            return render_template("index.html", prediction_text="Erro: O modelo não retornou uma previsão válida.")

        output = names[pred[0]]
        return render_template("index.html", prediction_text=f"Pokemon: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Erro ao processar a solicitação: {e}")



@app.route("/api", methods=["POST"])
def results():
    try:
        data = request.get_json(force=True)

        # Convert values to float safely
        features = []
        for value in data.values():
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({"error": "Todos os valores devem ser numéricos."}), 400

        pred = model.predict([np.array(features)])

        if len(pred) == 0:
            return jsonify({"error": "O modelo não retornou uma previsão válida."}), 500

        output = names[pred[0]]
        return jsonify({"prediction": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
