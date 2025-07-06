# Import required libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from heapq import nlargest

# Download necessary NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample text to summarize
text = """
The quick brown fox jumps over the lazy dog.
The fox is running fast to catch its prey. Suddenly, it sees the prey and jumps over it.
The prey escapes and the fox continues to run.
The dog wakes up and barks at the fox.
The fox runs away and the dog goes back to sleep.
"""

# Step 1: Tokenize text into sentences
sentences = sent_tokenize(text)

# Step 2: Remove stopwords and stem the words
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
words = []

for sentence in sentences:
    for word in word_tokenize(sentence):
        if word.lower() not in stop_words and word.isalpha():
            words.append(stemmer.stem(word))

# Step 3: Get word frequency
freq_dist = nltk.FreqDist(words)

# Step 4: Get top 10 most frequent words
top_words = [word for word, freq in freq_dist.most_common(10)]

# Step 5: Score sentences
summary = []
for sentence in sentences:
    sentence_words = word_tokenize(sentence.lower())
    sentence_score = 0
    for word in sentence_words:
        if stemmer.stem(word) in top_words:
            sentence_score += 1
    summary.append((sentence, sentence_score))

# Step 6: Print top 3 scored sentences as summary
print("Summary:\n")
for sentence in nlargest(3, summary, key=lambda x: x[1]):
    print(sentence[0])
