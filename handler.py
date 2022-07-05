from flask import Flask, request, Response
import pandas as pd
from top_bank import Top_bank
import os

# initialize API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def top_bank_predict():
    df_raw_json = request.get_json()
    
    if df_raw_json:
        if isinstance(df_raw_json, dict):
            df_raw = pd.DataFrame(df_raw_json, index=[0])
            
        else:
            df_raw = pd.DataFrame(df_raw_json, columns=df_raw_json[0].keys())
    
        # instantiate Top_bank_class
        papeline = Top_bank()

        # data_cleaning
        df = papeline.data_filtering(df_raw)

        # data_preparation
        df = papeline.data_preparation(df_raw)

        # get_propensity
        df = papeline.get_propensity(df, df_raw)

        # return json
        df = df.to_json(orient='records')
        return df
    
    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000) # heroku host
    app.run(host='0.0.0.0', port=port)