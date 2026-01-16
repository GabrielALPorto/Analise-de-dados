import pandas as pd
import pathlib

#workingPath = pathlib.Path(__file__).parent.resolve()
#dataPath = f'{workingPath}/dados_clients.csv' 
#print(dataPath)
# Venv instalou o Pandas no repositório "class_probability-estimation"!
csvDf = pd.read_csv('src/dados_clientes.csv')
print(csvDf.to_string())