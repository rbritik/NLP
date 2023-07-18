# Predicting POS(part of speech) tag for given word
Example 
| Sentence               | POS tag of word “back” |
|------------------------|-----------------------|
| The “back” door        | ADJECTIVE             |
| On my “back”           | NOUN                  |
| Win the voters “back”  | ADVERB                |
| Promised to “back” the bill | VERB              |

- In mostFreq, a simple trick of assiging most frequent tag for given word in dictionary is assigned to word from test dataset. For example 
   NOUN tag is associated most of times with "back" word, so this simple program will assign NOUN tag to "back" each time.

- In HMM, Viterbi algorithm based on dynammic programming is used to label POS for a entered text.
    Reference: <a href="https://en.wikipedia.org/wiki/Viterbi_algorithm#:~:text=The%20Viterbi%20algorithm%20is%20a,hidden%20Markov%20models%20(HMM)."> Viterbi Algorithm<a>
