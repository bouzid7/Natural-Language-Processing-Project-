import streamlit as st
import pandas as pd
from tqdm import tqdm
import joblib
import os 
import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

from transformers import MarianMTModel, MarianTokenizer, BertTokenizer
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, TFXLMRobertaModel
from transformers import AutoTokenizer, AutoConfig, TFAutoModel
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

language_list = ['de', 'fr', 'el', 'ja', 'ru']
data_path = "/content/data/Research_text_files.csv"
model_path = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

def preprocess_data(data_path, sample_size):
    # Read the data from a specific path
    data = pd.read_csv(data_path, low_memory=False)
    # Drop articles without Abstract
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    # Get "sample_size" random articles
    data = data.sample(sample_size)[['abstract', 'cord_uid']]
    return data

def create_vector_from_text(tokenizer, model, text, MAX_LEN = 510):

    input_ids = tokenizer.encode(
                        text,
                        add_special_tokens = True,
                        max_length = MAX_LEN,
                   )

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long",
                              truncating="post", padding="post")

    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks
    attention_mask = [int(i>0) for i in input_ids]

    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        logits, encoded_layers = model(
                                    input_ids = input_ids,
                                    token_type_ids = None,
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]

    # Extract the embedding.
    vector = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    vector = vector.detach().cpu().numpy()

    return(vector)

def translate_text(text, text_lang, target_lang='en'):
    # Get the name of the model
    model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
    # Get the tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Instantiate the model
    model = MarianMTModel.from_pretrained(model_name)
    # Translation of the text
    formated_text = ">>{}<< {}".format(text_lang, text)
    translation = model.generate(**tokenizer([formated_text], return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
    return translated_text

def process_document(text):
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)
    return text_vect

def is_plagiarism(similarity_score, plagiarism_threshold):
    is_plagiarism = False
    if similarity_score >= plagiarism_threshold:
        is_plagiarism = True
    return is_plagiarism

def check_incoming_document(incoming_document):
    text_lang = detect(incoming_document)
    language_list = ['de', 'fr', 'el', 'ja', 'ru']
    final_result = ""
    if text_lang == 'en':
        final_result = incoming_document
    elif text_lang not in language_list:
        final_result = None
    else:
        # Translate to English
        final_result = translate_text(incoming_document, text_lang)
    return final_result

def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3
    # Check the language of the query/incoming text and translate if required.
    document_translation = check_incoming_document(query_text)
    if document_translation is None:
        print("Only the following languages are supported: English, French, Russian, German, Greek, and Japanese")
        exit(-1)
    else:
        # Preprocess the document to get the required vector for similarity analysis
        query_vect = process_document(document_translation)
        # Run similarity Search
        data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
        data["similarity"] = data["similarity"].apply(lambda x: x[0][0])
        similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N + 1]
        formated_result = similar_articles[["abstract", "cord_uid", "similarity"]].reset_index(drop=True)
        similarity_score = formated_result.iloc[0]["similarity"]
        most_similar_article = formated_result.iloc[0]["abstract"]
        is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)
        plagiarism_decision = {'similarity_score': similarity_score,
                               'is_plagiarism': is_plagiarism_bool,
                               'most_similar_article': most_similar_article,
                               'article_submitted': query_text
                              }
        return plagiarism_decision

#source_data = preprocess_data(data_path, 100)
load_folder = "/content/vector_database_folder"
data_file_path = f"{load_folder}/data_with_vectors.joblib"
loaded_vector_database = joblib.load(data_file_path)
#vector_database = create_vector_database(source_data)
#new_incoming_text = source_data.iloc[1]['abstract']
#analysis_result = run_plagiarism_analysis(new_incoming_text,loaded_vector_database, plagiarism_threshold=0.8)
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    text_data = bytes_data.decode("utf-8")  # Decode bytes to text assuming UTF-8 encoding
    st.text(text_data)  # Display the text content
    analysis_result = run_plagiarism_analysis(text_data, loaded_vector_database, plagiarism_threshold=0.8)
    st.write(analysis_result)
else :
  st.write("wait...")

# st.write('# Display 5 samples:')
#st.write('# Display 5 samples:')
#st.write(source_data.sample(5))
#st.write('Run the plagiarism detection using an article from the database:')
#st.write(analysis_result)
