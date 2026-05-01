import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()
df = pd.DataFrame(data = digits.data, columns = digits.feature_names)
df['target'] = digits.target
