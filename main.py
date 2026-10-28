import pandas as pd

transactions = [
    ["Bread", "Butter", "Milk"],
    ["Bread", "Butter"],
    ["Bread", "Milk"],
    ["Butter", "Milk"],
    ["Bread", "Butter", "Milk"],
    ["Bread", "Eggs"],
    ["Milk", "Eggs"],
    ["Bread", "Butter", "Eggs"],
    ["Bread", "Milk", "Eggs"],
    ["Butter", "Milk"]
]

print("MARKET BASKET ANALYSIS")
print("----------------------")

for i, transaction in enumerate(transactions, start=1):
    print(f"Transaction {i}: {transaction}")

print("\nTotal Transactions:", len(transactions))
from mlxtend.preprocessing import TransactionEncoder

# Convert transactions into table format
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_data, columns=encoder.columns_)

print("\nTRANSACTION TABLE")
print(df.astype(int))
from mlxtend.frequent_patterns import apriori, association_rules

# Find frequently purchased item combinations
frequent_items = apriori(
    df,
    min_support=0.2,
    use_colnames=True
)

print("\nFREQUENT ITEMSETS")
print(frequent_items)
# Generate association rules
rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.5
)

print("\nASSOCIATION RULES")

if rules.empty:
    print("No association rules found.")
else:
    print(
        rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].to_string(index=False)
    )