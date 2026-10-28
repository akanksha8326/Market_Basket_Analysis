import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis")
st.subheader("DWDM Mini Project Using Data Mining and DevOps")

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

st.write("### Shopping Transactions")

for i, transaction in enumerate(transactions, start=1):
    st.write(f"Transaction {i}: {', '.join(transaction)}")

encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_data, columns=encoder.columns_)

st.write("### Transaction Data Table")
st.dataframe(df.astype(int), use_container_width=True)

st.write("### Frequently Purchased Items")

frequent_items = apriori(
    df,
    min_support=0.2,
    use_colnames=True
)

frequent_items["itemsets"] = frequent_items["itemsets"].apply(
    lambda items: ", ".join(sorted(items))
)

st.dataframe(frequent_items, use_container_width=True)

st.success("Market Basket Analysis completed successfully!")