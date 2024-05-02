## A naive autocorrection program
- words n-edit distance away from entered word will be searched and among those words word with highest probability will be suggested
- Vocabulary is made from mini shakespeare dataset.
- Example:
      Entered word: dys  
      suggestions =  {'days', 'dye'}  
      word 0: days, probability 0.000410  
      word 1: dye, probability 0.000019  
- reference <a href="https://norvig.com/spell-correct.html">spell-correct</a>
