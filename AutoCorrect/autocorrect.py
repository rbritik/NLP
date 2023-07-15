import re
from collections import Counter
import numpy as np
import pandas as pd


def process_data(file_name):
    """
    Input: 
        file_name: A file_name
    Output:
        words:  a list containing all the words in the corpus in lower case
    """
    
    with open(file_name) as f:
        text = f.read()
    
    text = text.lower()
    words = re.findall(r'\w+', text)
    return words


def get_count(word_l):
    """
    Input:
        word_l: a set of words representing the corpus
    Output:
        word_count_dict: The wordcount dicionary where key is the word and value is its frequency 
    """
    word_count_dict = {}
    for word in word_l:
        word_count_dict[word] = word_count_dict.get(word, 0) + 1
        
    return word_count_dict


def get_probs(word_count_dict):
    """
    Input: 
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur 
    """
    probs = {}
    N = sum(word_count_dict.values())
    
    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word]/N
        
    return probs


def delete_letter(word):
    """
    Input:
        word: string
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    """
    
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    delete_l = [L + R[1:] for L, R in split_l if R]
    
    return delete_l


def switch_letter(word):
    """
    Input:
        word: input string
    Output: a list of all possible strings with one adjacent character switched 
    """
    
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    switch_l = [L[:-1] + R[0] + L[-1] + R[1:] for L, R in split_l if L]
    
    return switch_l

def replace_letter(word):
    """
    Input: 
        word: the input string
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word. 
    """
    
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l = [L + l + R[1:] for l in letters for L, R in split_l]
    replace_set = set(replace_l)
    replace_set.remove(word)
    
    replace_l = sorted(list(replace_set))
    
    return replace_l


def insert_letter(word):
    """ 
    Input:
        word: the input string
    Output:
        inserts: a set of all possible srings with one new letter inserted at every offset.
    """
    
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insert_l = [L + l + R for l in letters for L, R in split_l]
    
    return insert_l


def edit_one_letter(word, allow_switches = True):
    """
    Input:
        word: string
    Output:
        edit_one_set: a set of words with one possible edit away from input string.
    """
    
    delete_l = delete_letter(word)
    replace_l = replace_letter(word)
    insert_l = insert_letter(word)
    switch_l = []
    if allow_switches:
        switch_l = switch_letter(word)
    
    edit_one_set = set(delete_l + replace_l + insert_l + switch_l)
    
    return edit_one_set


def edit_two_letters(word, allow_switches = True):
    """
    Input:
        word: the input string
    Output:
        edit_two_set: a set of strings with all possible two edits
    """
    
    edit_two_set = set()
    
    edit_one_set = edit_one_letter(word, allow_switches)
    for w in edit_one_set:
        if w:
            edit_one = edit_one_letter(w, allow_switches)
            edit_two_set.update(edit_one)
    
    return edit_two_set


def get_corrections(word, probs, vocab, n=2, verbose=False):
    """
    Input: 
        word: a user entered string to check for suggestions.
        probs: a dictionary that maps each word to its probability.
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best = a list of tuples with the most probable n corrected words and their probabilities
    """ 
    
    suggestions = set(list((word in vocab and word) or vocab.intersection(edit_one_letter(word)) or vocab.intersection(edit_two_letters(word))))
    
    sugg_prob = sorted([(w, probs[w]) for w in suggestions], key = lambda x: x[-1], reverse=True)
    n_best = sugg_prob[:n]
    
    if verbose:
        print("entered word = ", word, "\nsuggestions = ", suggestions)
        
    return n_best
    
    
    

if __name__ == "__main__":
    word_l = process_data('./data/shakespeare.txt')
    vocab = set(word_l)
    print("Size of vocabulary: ", len(vocab))
    
    # get word count of each word from vocabulary
    word_count_dict = get_count(word_l)
    
    # probability of each word to be in corpus
    probs = get_probs(word_count_dict)
    
    my_word = 'dys'
    tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose = True)
    
    for i, word_prob in enumerate(tmp_corrections):
        print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")   
    
    
    
    
