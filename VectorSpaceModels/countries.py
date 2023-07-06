import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# First five elements in the dataframe 
print(data.head(5))

# Each of the word embedding is a 300-dimensional vector
word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb")) # A subset of word2vec embeddings


def cosine_similarity(A, B):
    """
    Args:
        A (numpy array): A numpy array which corresponds to a word vector
        B (numpy array):  A numpy array which corresponds to a word vector
        
    Output:
        cos: numerical number representing the cosine similarity between A and B
    """
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    
    cos = dot / (norma * normb)
    
    return cos

def euclidean(A, B):
    """
    Args:
        A (numpy array): A numpy array which corresponds to a word vector
        B (numpy array):  A numpy array which corresponds to a word vector
        
    Output:
        d: numerical number representing the Euclidean distance between A and B
    """
    d = np.linalg.norm(A - B)
    return d

def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    """
    Args:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their emmbeddings
    Output:
        countries: a tuple with the most likely country and its similarity score
    """
    
    group = {city1, country1, city2}
    
    city1_emb = embeddings[city1]
    city2_emb = embeddings[city2]
    country1_emb = embeddings[country1]
    
    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # King - Man + Woman = Queen
    vec = country1_emb - city1_emb + city2_emb
    
    similarity = -1
    country = ""
    
    for word in embeddings.keys():
        if word not in group:
            word_emb = embeddings[word]
            cur_similarity = cosine_similarity(vec, word_emb)
            
            if cur_similarity > similarity:
                similarity = cur_similarity
                country = (word, similarity)
    return country

print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))

def get_accuracy(word_embeddings, data, get_country=get_country):
    '''
    Args:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas DataFrame containing all the country and capital city pairs
    Output:
        accuracy: correct calculated / total examples
    '''
    
    num_correct = 0
    
    for i, row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']
        
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)
        
        # if the predicted country2 is same as the actual country2
        if predicted_country2 == country2:
             num_correct += 1
    
    m = len(data)
    accuracy = num_correct / m
    
    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")   # 0.92 accuracy

    
    