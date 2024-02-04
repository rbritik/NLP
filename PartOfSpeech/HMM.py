"""
    POS tagging using Hidden Markov Model
"""
import numpy as np
import math
from collections import defaultdict
from utils import build_vocab, preprocess, get_word_tag
from mostFreqPOS import create_dictionaries 


def create_transition_matrix(alpha, tag_counts, transition_counts):
    """ 
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping where each tag to its repspective count
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
    Output:
        A: matrix of dimension (num_tags, num_tags)
    """
    
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    
    A = np.zeros((num_tags, num_tags))
    
    trans_keys = set(transition_counts.keys())
    
    for i in range(num_tags):
        for j in range(num_tags):
            
            # count of two tags appearing together
            count = 0
            key = (all_tags[i], all_tags[j])
            
            # check whether key is in dictionary
            if key in transition_counts:
                count = transition_counts[key]
                
            count_prev_tag = tag_counts[key[0]]
            
            # Probability of transitioning from ith tag to jth tag
            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)
            
    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    """ 
    Input:
        alpha: tunning parameter used in smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    """
    
    all_tags = sorted(tag_counts.keys())
    num_tags = len(tag_counts)
    
    num_words = len(vocab)
    
    B = np.zeros((num_tags, num_words))
    
    emis_keys = set(list(emission_counts.keys()))
    
    for i in range(num_tags):
        for j in range(num_words):
            
            # count of jth word appearing with ith tag tag
            count = 0
            
            key = (all_tags[i], vocab[j])
            if key in emis_keys:
                count = emission_counts[key]
                
            count_tag = tag_counts[key[0]]
            
            # Probability of word with ith tag is jth 
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
            
    return B
    

# -------------Veterbi Algorith and Dynammic Programming ---------------- #
#   3 steps
#   1. intialization - intialize best_paths and bes_probabiltity matrices
#   2. feed forward - At each step the probabilityof each path happening and the best paths up to that point is calculated
#   3. feed backward - ALlows to find the best path with the highest probability

def initialize(states, tag_counts, A, B, corpus, vocab):
    """ 
    Input:
        states: a list of all possible part-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(copus)) of integers
    """ 
    
    num_tags = len(tag_counts)
    
    # POS tags in the rows, number of words in the copus as the columns
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    
    s_idx = states.index('--s--')   # start token
    
    for i in range(num_tags):
        # transitioning from starting tag to it tag and given probability of first word from corpus for ith tag
        best_probs[i, 0] = math.log(A[s_idx, 1]) + math.log(B[i, vocab[corpus[0]]])

    return best_probs, best_paths

def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    """ 
    Input:
         A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initialized matrix of dimension (num_tags, len(corpus))
        best_paths: an initialized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index 
    Output: 
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    """
    
    num_tags = best_probs.shape[0]
    
    # for every word in the corpus starting from word 1
    for i in range(1, len(test_corpus)):
        
        # for each unique POS tag that the current word can be 
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None
            
            # for each POS tag that the previous word can be
            for k in range(num_tags):
                
                prob = best_probs[k][i-1] + math.log(A[k][j]) + math.log(B[j][vocab[test_corpus[i]]])
                
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i
    
    return best_probs, best_paths 

def viterbi_backward(best_probs, best_paths, corpus, states):
    """
    Input: 
        best_probs: an initialized matrix of dimension (num_tags, len(corpus))
        best_paths: an initialized matrix of dimension (num_tags, len(corpus))
        corpus: a list containing a preprocessed corpus
        states: a list conatining different tags 
    Output:
        pred: a list containing the best tags for each word in corpus 
    """
    
    m = best_paths.shape[1]
    
    num_tags = best_probs.shape[0]  # Number of unique POS tags
    
    best_prob_for_last_word = float('-inf')
    
    # lists containing index and string representation of best tag for each word in corpus, same length as corpus
    z = [None] * m 
    pred = [None]*m
    
    # find POS tag with highest prob for last word of the corpus
    for k in range(num_tags):
        
        if best_probs[k][m-1] > best_prob_for_last_word:
            best_prob_last_word = best_probs[k][m-1]
            z[m - 1] = k
    
    # predicted tag's index for last word
    pred[m - 1] = states[z[m-1]]
    
    for i in range(m - 1, 0, -1):
        
        pos_tag_for_word_i = z[i] 
        z[i - 1] = best_paths[pos_tag_for_word_i][i]
        pred[i - 1] = states[z[i - 1]]
    
    return pred
    
    
def compute_accuracy(pred, y):  
    """ 
    nput: 
        pred: a list of the predicted parts-of-speech 
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:
        accuracy: correctly_predicted / total 
    """
    
    num_correct = 0
    total = 0
    
    for prediction, y in zip(pred, y):
        # label splitted into the word and POS tag
        word_tag_tuple = y.split()
        
        if (len(word_tag_tuple) != 2):
            continue
        
        word, tag = word_tag_tuple
        
        if tag == prediction:
            num_correct += 1
        
        total += 1
        
    accuracy = num_correct / total
    
    return accuracy
    


if __name__ == "__main__":
    
    # Load the training corpus
    with open("./data/WSJ_02-21.pos", "r") as f:
        training_corpus = f.readlines()
        
    # Build vocabulary
    # voc_l = build_vocab('./data/WSJ_02-21.pos') 
    with open("./data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')
    vocab = {}
    for i, word in enumerate(sorted(voc_l)):
        vocab[word] = i
    
    # load the test corpus
    with open("./data/WSJ_24.pos", "r") as f:
        y = f.readlines()
        
    _, prep = preprocess(vocab, "./data/test.words")
    
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab, False)
    
    # get all the POS states
    states = sorted(tag_counts.keys())
    
    # For smoothing transition and emission matrices
    alpha = 0.001
    # Prepare transition_matrix
    A = create_transition_matrix(alpha, tag_counts, transition_counts)
    # Prepare emission_matrix
    B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))
    
    # Viterbi's alogrithm 
    best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
    print("running")
    best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
    print("running")
    pred = viterbi_backward(best_probs, best_paths, prep, states)
    m = len(pred)
    print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
    print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])
    
    # predicting on a dataset
    print('The third word is:', prep[3])
    print('prediction is:', pred[3])
    print('corresponding label y is: ', y[3])
    
    print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")
    # Accuracy achieved is about 93% that is better than simple mostFrePOS method
