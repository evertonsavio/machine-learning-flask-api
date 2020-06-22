from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import pickle
import json
##############################################################################

app = Flask(__name__, static_folder='static')

@app.route('/') # '/' e a homepage da aplicação.
def home():
    return render_template('index.html')

@app.route('/model', methods=['POST'])
def create_model():
    request_data = request.get_json() # get_json converte json para dictionaty.
    df = pd.DataFrame(json.loads(request_data["payload"]))

    reg = pickle.load(open('static/model.pkl','rb'))
    preprocessed_data = pd.DataFrame()
    data = df.copy()

    if (data is not None):
        preprocessed_data['Probabilidade_nao_tem_doenca'] = 0
        preprocessed_data['Probabilidade_tem_doenca'] = 0
        preprocessed_data['Predito'] = 0

        preprocessed_data[['Probabilidade_nao_tem_doenca', 'Probabilidade_tem_doenca']] = reg.predict_proba(data)
        preprocessed_data['Predito'] = reg.predict(data)

    df_to_dict_data = preprocessed_data.to_json(index=False, orient='table')

    return jsonify(df_to_dict_data)

@app.route('/test', methods=['POST'])
def testar_rota():
    return jsonify({"Ola": "Sucesso!"})

if __name__ == "__main__":
    app.run(port=5000)

###############################################################################
# Back to the basics:
# Isso é um servidor backend entao quando recebemos uma requisição do tipo:
# post: estamos recebendo dados.
# get: enviando dados.
# no browser é o contrario.
