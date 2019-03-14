import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk, ngrams
from nltk import FreqDist
import collections 
import re
import string


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
ps = PorterStemmer() # used as a variable to call Porterstemmmer
lemmatizer = WordNetLemmatizer() # used as a variable to call lemmatizer of word

#reading data from file
with open('nlp_input.txt', 'r',encoding="utf8", errors='ignore') as f:
    contents = f.read().strip()
    
contents = contents.replace('??', 'qq')
contents = contents.replace('(?)', '(q)')

contents = contents.replace('Setosa.io)', 'Setosa.io).')
contents = contents.replace('Equation for least ordinary squares', 'Equation for least ordinary squares.')
contents = contents.replace('These are known as L1 regularization(Lasso regression) and L2 regularization(ridge regression).',
                    'These are known as L1 regularization(Lasso regression) and L2 regularization(ridge regression). ')
contents = contents.replace('L2 regularization penalty term', 'L2 regularization penalty term.')
contents = contents.replace('n(', 'n (')
contents = contents.replace('L1 regularization penalty term', 'L1 regularization penalty term.')
contents = contents.replace('significantly higher weight than the rest', 'significantly higher weight than the rest.')
contents = contents.replace('Performing Lasso regression', 'Performing Lasso regression.')
contents = contents.replace('Performing Elastic Net regression', 'Performing Elastic Net regression.')
contents = contents.replace('1 denotes lasso)', '1 denotes lasso).')
  

#Sentence tonization and work tokenization
stokens = nltk.sent_tokenize(contents)#sentence tokenizing
wtokens = nltk.word_tokenize(contents)# word tokenizing
print(wtokens)
# lemmatizing the data
print("lemmatizing:")

lem=[]
for w in wtokens:
    lem.append(lemmatizer.lemmatize(w))

print(lem)


# performing Trigram on the  Lemmatizer Output
print("Trigrams :\n")
TrigramsOutput = []
for top in ngrams(wtokens, 3):
    # Fetching Trigrams using 'ngrams' method and Iterating it and appending it to list
    TrigramsOutput.append(top)
print(TrigramsOutput)



# TriGram- Word Frequency
# Using TrigramOutput fetch the WordFreq Details
wordFreq = FreqDist(TrigramsOutput)
# Most commoon Trigrams
mostCommon = wordFreq.most_common()
print("TriGrams Frequency in asecing order : \n", mostCommon)

# First 10 Trigrams
First10 = wordFreq.most_common(10)
print("First 10 Trigrams : \n", First10)

Trigram = []

for k,v in mostCommon:
    s = ' '.join(k)
    Trigram.append(s)

filecontents = ''

for t in Trigram:
    for s in stokens:
        if t in s:
            filecontents = filecontents + s

print('\n the concatinated result is:\n',filecontents)
