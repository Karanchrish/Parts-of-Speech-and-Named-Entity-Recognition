import streamlit as st
import spacy

nlp_pos = spacy.load("en_core_web_sm")
nlp_ner = spacy.load("en_core_web_sm")

def pos_tagging(text):
    doc = nlp_pos(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def named_entity_recognition(text):
    doc = nlp_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

st.title("Parts Of Speech and Named Entity Recognition Tool")

st.header("Input Text")
input_text = st.text_area("Enter your text :")

task = st.sidebar.radio("Select Task:", ("Parts Of Speech Tagging", "Named Entity Recognition"))

if st.button("Perform Task"):
    if task == "Parts Of Speech Tagging":
        pos_tags = pos_tagging(input_text)
        st.header("Parts Of Speech Tagging Results")
        st.table(pos_tags)
    else:
        entities = named_entity_recognition(input_text)
        st.header("Named Entity Recognition Results")
        st.table(entities)
