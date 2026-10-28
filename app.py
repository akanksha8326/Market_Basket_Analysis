import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Basket Insights", page_icon="🛍️", layout="wide")
st.markdown("<h1 style='text-align:center;color:#7137c8'>🛍️ Basket Insights</h1><p style='text-align:center'>Discover what customers buy together</p>", unsafe_allow_html=True)

DEMO = [["Bread","Milk","Butter"],["Bread","Diapers","Beer","Eggs"],["Milk","Diapers","Beer","Cola"],["Bread","Milk","Diapers","Beer"],["Bread","Milk","Diapers","Cola"],["Bread","Milk","Eggs"],["Milk","Butter"],["Bread","Butter","Eggs"],["Milk","Diapers","Butter"],["Bread","Milk","Diapers","Beer"]]

def load_data(file):
    if file is None: return DEMO, "Demo dataset"
    df = pd.read_csv(file)
    cols = {c.lower().replace("_","").replace(" ",""):c for c in df.columns}
    tid = next((cols[k] for k in ["transactionid","transaction","orderid","invoiceno","billid"] if k in cols), None)
    item = next((cols[k] for k in ["item","itemname","itemdescription","product","productname"] if k in cols), None)
    if not tid or not item: raise ValueError("CSV needs transaction_id and item columns. Download the sample to see the format.")
    df = df[[tid,item]].dropna(); df[item] = df[item].astype(str).str.strip()
    return df.groupby(tid, sort=False)[item].agg(lambda x: sorted(set(x))).tolist(), file.name

with st.sidebar:
    st.header("🎛️ Analysis controls")
    file = st.file_uploader("Upload transaction CSV", type="csv")
    support = st.slider("Minimum support", .05, .8, .2, .05)
    confidence = st.slider("Minimum confidence", .1, 1., .5, .05)
    max_len = st.slider("Largest combination", 2, 5, 3)
    csv = pd.DataFrame([(i+1,x) for i,b in enumerate(DEMO) for x in b], columns=["transaction_id","item"]).to_csv(index=False).encode()
    st.download_button("⬇️ Download sample CSV", csv, "sample_transactions.csv", "text/csv", use_container_width=True)
try:
    transactions, source = load_data(file)
except Exception as e:
    st.error(str(e)); st.stop()
if len(transactions)<2: st.warning("Please provide at least two transactions."); st.stop()
enc = TransactionEncoder(); basket = pd.DataFrame(enc.fit(transactions).transform(transactions), columns=enc.columns_)
sets = apriori(basket, min_support=support, use_colnames=True, max_len=max_len)
rules = association_rules(sets, metric="confidence", min_threshold=confidence) if not sets.empty and any(sets.itemsets.map(len)>=2) else pd.DataFrame()
products = pd.DataFrame({"Product combination":[" + ".join(sorted(s)) for s in sets.itemsets],"Support (%)":(sets.support*100).round(1)}) if not sets.empty else pd.DataFrame()
if not rules.empty:
    rules = rules.assign(
        **{"If customers buy":rules.antecedents.map(lambda s:" + ".join(sorted(s))),"They may also buy":rules.consequents.map(lambda s:" + ".join(sorted(s))),"Support (%)":(rules.support*100).round(1),"Confidence (%)":(rules.confidence*100).round(1),"Lift":rules.lift.round(2)})
    rules = rules[["If customers buy","They may also buy","Support (%)","Confidence (%)","Lift"]].sort_values("Lift",ascending=False)
metrics=st.columns(4)
for col,label,value in zip(metrics,["Transactions","Products","Frequent patterns","Rules"],[len(transactions),len(basket.columns),len(products),len(rules)]): col.metric(label,value)
st.caption(f"Source: {source}  ·  Support: {support:.0%}  ·  Confidence: {confidence:.0%}")
t1,t2,t3=st.tabs(["📊 Overview","🧩 Patterns & rules","📘 Project explanation"])
with t1:
    a,b=st.columns(2)
    with a:
        st.subheader("Most popular products"); st.bar_chart(basket.astype(int).sum().sort_values(ascending=False).head(10))
    with b:
        st.subheader("Your transactions"); st.dataframe(pd.DataFrame({"Basket":[", ".join(x) for x in transactions]}),use_container_width=True,hide_index=True)
with t2:
    st.subheader("Frequent combinations")
    if products.empty: st.info("No patterns at this support. Lower minimum support in the sidebar.")
    else:
        st.dataframe(products.sort_values("Support (%)",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Download patterns CSV",products.to_csv(index=False).encode(),"frequent_patterns.csv","text/csv")
    st.subheader("Recommendations (association rules)")
    if rules.empty: st.info("No rules yet. Lower support or confidence to discover more.")
    else:
        st.dataframe(rules,use_container_width=True,hide_index=True)
        st.download_button("Download rules CSV",rules.to_csv(index=False).encode(),"association_rules.csv","text/csv")
with t3:
    st.subheader("How this project works")
    st.markdown("1. **Support**: how often products appear together.\n2. **Confidence**: how often the suggested product is bought when the first product is bought.\n3. **Lift**: values above 1 suggest a stronger-than-average relationship.\n\nThis project uses the **Apriori algorithm** to find frequent item combinations and create association rules. Try the demo first, then upload a CSV with `transaction_id` and `item` columns.")
st.markdown("<hr><center>Built with Streamlit · Apriori · Association Rule Mining</center>",unsafe_allow_html=True)
