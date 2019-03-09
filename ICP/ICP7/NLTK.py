from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import wordpunct_tokenize,ne_chunk,pos_tag
from bs4.element import Comment
import urllib.request

nltk.download('maxent_ne_chunker')


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Google').read()

print(text_from_html(html))

# Writing text into txt file

file = open('input.txt', 'w')

file.write(text_from_html(html))

file.close()


with open('input.txt', 'r') as file:
    fileread = file.read()


#Sentence tonization and work tokenization
senttokens = nltk.sent_tokenize(fileread)
wordtokens = nltk.word_tokenize(fileread)

#printing each sentence
for s in senttokens:
    print(s)

#printing each word
for w in wordtokens:
    print(w)


#Porter Stemmer
pStemmer=PorterStemmer()
print("Porter steeming output \n")
for p in wordtokens:
    print(pStemmer.stem(str(p)))

#lancasters Stemmer
lStemmer=LancasterStemmer()
print(" Lancaster stemming output\n")
for t in wordtokens:
    print(lStemmer.stem(str(t)))

#Snowball Stemmer
sStemmer = SnowballStemmer('english')
print("Snowball steeming output \n")
for s in wordtokens:
    print(sStemmer.stem(str(s)))

#parts of speech
print("Parts of Speech \n")
print(nltk.pos_tag(wordtokens))

#Lemmatizer
print("Lemmatizer \n")
lemmatizer = WordNetLemmatizer()
for l in wordtokens:
    print(lemmatizer.lemmatize(str(l)))

#Trigram
print("Trigram \n")
trigram = ngrams(wordtokens, 3)
for t in trigram:
    print(t)


#Named Entity Recognition
print("Named Entity Recognition")
for s in senttokens:
    ner = ne_chunk(pos_tag(wordpunct_tokenize(s)))
    print(str(ner))
