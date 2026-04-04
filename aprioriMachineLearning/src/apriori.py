import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Example by: https://www.geeksforgeeks.org/machine-learning/implementing-apriori-algorithm-in-python/

# Importing the groceries_dataset
df = pd.read_csv('aprioriMachineLearning/src/Groceries_dataset.csv')
#print(df.head())

# grouping items purchased together
# groupby gets a combination of splitting objects, apply a function and combine the results
# apply use a function along an axis of the DataFrame
# reset_index create a new column with an id to be the new index
basket = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
#print(basket)
# gets the column with the label "itemDescription" and transform to a list
transactions = basket['itemDescription'].tolist()
#print(transactions)

# transforms the data into a matrix containing True or False
transacEncoder = TransactionEncoder()
# fit learns the labels of the dataset
# meanwhile, transform turns the data into "True" or "False" utilizing the labels
transacEncoderArray = transacEncoder.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(transacEncoderArray, columns = transacEncoder.columns_)

# the apriori finally will be used. We need to choose a minimum support value
# our min support value will be 0.01. It means that items bought together in a rate of
# 1% will be in our itemsets
# apriori thinking: "if an itemset is frequent, then all its subsets will be frequent too"
# only itemsets with more than 1% of products bought together will be show
frequent_itemsets = apriori(df_encoded, min_support = 0.01, use_colnames = True)
#print(f"Total Frequent Itemsets: {frequent_itemsets.shape[0]}")

# generating association rules
# support: how often the rule appears in the dataset
# confidence: probability of buying item B if item A is bought
# lift: strength of the rule over random chance. (>1 means a good rule)
rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.1)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1) & rules['consequents'].apply(lambda x: len(x) >= 1)]
#print(f'Association Rules: {rules.shape[0]}')
#print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
# all results have lift < 1, so the subsets of products inhibits each other
# the products shouldn't be together in a supermarket!

# visualizing the most popular items
top_items = df['itemDescription'].value_counts().head(10)
top_items.plot(kind = 'bar', title = 'The top 10 Most Purchased Items')
plt.xlabel("Item")
plt.ylabel("Count")
plt.show()
