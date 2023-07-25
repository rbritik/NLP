import math
import random
import numpy as np
from transformers import AutoTokenizer
from utils import get_tokenized_data, preprocess_data


def count_n_grams(data, n, start_token = '<s>', end_token = '<e>'):
    """
    Count all n-grams in the data
    Args:
        data: List of lists of words
        n: number of words in a sequence
    Returns:
        A dictionary  that maps a tuplle of n-words to its frequency
    """
    
    n_grams = {}
    
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)  # key in the dictionary
        
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i: i+n]
            
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    
    return n_grams


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    Returns:
        A probability
    """
    previous_n_gram = tuple(previous_n_gram)
    
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    
    numerator = n_plus1_gram_count + k
    
    probability = numerator / denominator
    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token='<unk>', k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    """    
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts,
                                           n_plus1_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability
    
    return probabilities
    
    
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):
    """
    Get suggestion for the next word
    Args: 
        previous_tokens: The sentence you input where each token is a word. Must have length >= n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word
    Returns:
        A tuple of 
            - string of the most likely next word
            - corresponding probability
    """
    # length of previous word
    n = len(list(n_gram_counts.keys())[0])
    
    # append start token on previous tokens
    previous_tokens = ['<s>'] * n + previous_tokens
    
    previous_n_gram = previous_tokens[-n:]  # most recent 'n' words as the previous n-gram
    
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                            vocabulary, k=k)
    
    suggestion = None
    max_prob = 0
    
    for word, prob in probabilities.items():
        if start_with is not None:
            # check if the begining of words does not match with the letters in 'start_with'
            if not word.startswith(start_with):
                continue
        
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    
    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k = 1.0, start_with=None):
    """ Generate a word for different n-gram models
    Args:
        previous_tokens: previous tokens in the sentence
        n_gram_counts_list: list for different n-grams related models
        vocabulary: list of unique words
        k: constant, smoothing parameter
    Returns:
        list of tuples of next word for different n-grams (n can be 1, 2, 3,...)
    """
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, 
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    
    return suggestions


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token='<e>', k=1.0):
    """
    Calculate perplexity for a list of sentences
    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant
    Returns:
        Perplexity score (how closer it to the human generated sentence)
    """    
    n = len(list(n_gram_counts.keys())[0])
    
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)
    
    product_pi = 1.0
    for t in range(n, N):
        n_gram = sentence[t-n: t]
        word = sentence[t]
        
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0)
        
        product_pi *= 1/probability 
        
    perplexity = (product_pi)**(1/N)
    return perplexity


if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # bert base tokenizer
    
    # read data
    with open('./en_US.twitter.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    print("Number of letters:", len(data))
    
    # get tokenized data
    tokenized_data = get_tokenized_data(data, tokenizer)
    
    
    # split data into training set and test set
    random.seed(84)
    random.shuffle(tokenized_data)
    
    train_size = int(len(tokenized_data) * 0.8)
    
    train_data = tokenized_data[0: train_size]
    test_data = tokenized_data[train_size:]
    
    print("{} data are split into {} train and {} test set".format(len(tokenized_data), len(train_data), len(test_data)))
    
    minimum_freq = 2  # atleast word should be twice in dataset or it will be marked as <unk>
    train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, minimum_freq)
    
    print("First preprocessed training sample:")
    print(train_data_processed[0])
    print()
    
    print("First preprocessed test sample:")
    print(test_data_processed[0])
    print()
    
    # test the model
    sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = count_n_grams(sentences, 1)
    bigram_counts = count_n_grams(sentences, 2)
    perplexity_train = calculate_perplexity(sentences[0],
                                         unigram_counts, bigram_counts,
                                         len(unique_words), k=1.0)
    print(f"Perplexity for first train sample: {perplexity_train:.4f}")

    test_sentence = ['i', 'like', 'a', 'dog']
    perplexity_test = calculate_perplexity(test_sentence,
                                        unigram_counts, bigram_counts,
                                        len(unique_words), k=1.0)
    print(f"Perplexity for test sample: {perplexity_test:.4f}")
    
    previous_tokens = ['i', 'like']
    tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
    print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")
    
    print()
    # when setting the starts_with
    tmp_starts_with = 'c'
    tmp_suggest2 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
    print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")
    
    # For 1 to 5-grams getting suggestion for next word
    n_gram_counts_list = []
    for n in range(1, 6):
        print("Computing n-gram counts with n =", n, "...")
        n_model_counts = count_n_grams(train_data_processed, n)
        n_gram_counts_list.append(n_model_counts)
        
    previous_tokens = ["i", "am", "to"]
    tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

    print(f"The previous words are {previous_tokens}, the suggestions are:")
    print(tmp_suggest4)