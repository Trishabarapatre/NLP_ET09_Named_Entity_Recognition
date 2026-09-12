import streamlit as st
import spacy
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Page setup
st.set_page_config(page_title="NER Extractor", layout="wide")
st.title("🔍 Named Entity Recognition (NER) Tool")
st.write("Enter any text below to automatically extract named entities like People, Organizations, Locations, Dates, and more.")

# Text input
user_text = st.text_area("Enter your text here:", height=150, 
    placeholder="Example: Elon Musk founded SpaceX in California in 2002.")

# Button to analyze
if st.button("Extract Entities"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        doc = nlp(user_text)
        
        if len(doc.ents) == 0:
            st.info("No named entities found in this text.")
        else:
            # Visual highlighting
            st.subheader("Highlighted Entities")
            html = displacy.render(doc, style="ent")
            st.markdown(html, unsafe_allow_html=True)
            
            # Table of entities
            st.subheader("Entity Details")
            entities_data = []
            for ent in doc.ents:
                entities_data.append({
                    "Entity": ent.text,
                    "Type": ent.label_,
                    "Description": spacy.explain(ent.label_)
                })
            st.table(entities_data)

st.markdown("---")
st.caption("Built using spaCy | NLP Project - Named Entity Recognition")