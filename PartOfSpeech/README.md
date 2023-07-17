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
