from collections import defaultdict
from utils import build_vocab, preprocess, get_word_tag

def create_dictionaries(training_corpus, vocab, verbose = True):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is and index
        
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """
    
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    i = 0
    
    prev_tag = '--s--'  # start state
    
    for word_tag in training_corpus:
        
        i += 1
        if i % 50000 == 0 and verbose:
            print(f"word count = {i}")
        word, tag = get_word_tag(word_tag, vocab)
        
        transition_counts[(prev_tag, tag)] += 1
        
        emission_counts[(tag, word)] += 1
    
        tag_counts[tag] += 1
        
        prev_tag = tag
        
    return emission_counts, transition_counts, tag_counts    
  
    
def predict_pos(prep, y, emission_counts, vocab, states):
    """ 
    Input: 
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag, word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags
        
    Output:
        accuracy: Number of times a word is correctly classified
    """  
    # Prediction for tag is done only basis of most frequent tag for given word
    
    num_correct = 0
    
    all_words = set(emission_counts.keys())
    
    total = 0
    for word, y_tup in zip(prep, y):
        
        # split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()
        
        if len(y_tup_l) == 2:
            true_label = y_tup_l[1]
            
        else:
            continue
        
        count_final = 0
        pos_final = ''
        
        # If the word is in the vocabulary
        if word in vocab:
            for pos in states:
                
                key = (pos, word)
                
                if key in emission_counts:
                    count = emission_counts[key]
                    
                    if count > count_final:
                        # update the final count (largest count)
                        count_final = count
                        pos_final = pos
                        
            if pos_final == true_label:
                num_correct += 1
        
        total += 1
        
    accuracy = num_correct / total
    
    return accuracy
                   

if __name__ == "__main__":
    
    # Load the training corpus
    with open("./data/WSJ_02-21.pos", "r") as f:
        training_corpus = f.readlines()
        
    # Build vocabulary
    voc_l = build_vocab('./data/WSJ_02-21.pos') 
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
    
    # testing
    accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
    print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")