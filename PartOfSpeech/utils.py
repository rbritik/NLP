import string
from collections import defaultdict


def assign_unk(word):
    """ 
    Assign tokens to unknown words
    """
    
    punct = set(string.punctuation)
    
    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]
    
    # check whether any character in the word is a digit
    if any(char.isdigit() for char in word):
        return "--unk_digit--"
    
    # check whether any character in the word is a punctuation
    elif any(char in punct for char in word):
        return "--unk_punct--"
    
    # check whether any character in the word is uppercase
    elif any(char.isupper() for char in word):
        return "--unk_upper--"
    
    # check whether the word ends like frequently used nouns
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    
    # check whether the word ends like frequently used verbs
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    
    # check whether the word ends like frequently used adjectives
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    
    # check whether the word ends like frequently used adverbs
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    
    # If none of the criteria is met
    return "--unk--"


def get_word_tag(line, vocab):
    
    # If line is empty return placeholders for word and tag
    if not line.split():
        word = "--n--"
        tag = "--s--"
    
    else:
        word, tag = line.split()
        # if word is not in vocabulary
        if word not in vocab:
            # Handle unknown word
            tag = assign_unk(word)
    
    return word, tag
    
    
def build_vocab(datapath):
    with open(datapath, 'r') as f:
        lines = f.readlines()
        
    words = [line.split('\t')[0] for line in lines]
    
    freq = defaultdict(int)
    
    # count frequency of occurence for each word in the dataset
    for word in words:
        freq[word] += 1
    
    vocab = [k for k, v in freq.items() if (v > 1 and k != '\n')]
    vocab.sort()
    
    return vocab
    
def preprocess(vocab, data_fp):
    """ 
    Preprocess data
    """
    orig = []
    prep = []
    
    # Read data
    with open(data_fp, "r") as data_file:
        
        for cnt, word in enumerate(data_file):
            
            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue
            
            # Hadnle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue
            
            else:
                orig.append(word.strip())
                prep.append(word.strip())
    
    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))
    
    return orig, prep