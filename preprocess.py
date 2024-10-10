# -*- coding: utf-8 -*-
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy

class TextPreprocessor:
    def __init__(self):
        """Initialize the TextPreprocessor with English stop words and Spacy model."""
        self.stop_words = set(stopwords.words('english'))  # Set of English stop words
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # Load Spacy model
        self.mapping = {}  # Map to store word to index mapping
        self.n_components = 0  # Count of unique words

    def sent_to_words(self, sentences):
        """Convert a list of sentences into a list of tokenized words."""
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    def remove_stopwords(self, texts):
        """Remove stopwords from a list of texts."""
        return [
            [word for word in simple_preprocess(str(doc)) if word not in self.stop_words]
            for doc in texts
        ]

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Lemmatize the words in the texts while filtering by allowed parts of speech."""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent))  # Process the sentence with Spacy
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])  # Lemmatize
        return texts_out

    def check_words(self, text_collection):
        """Organize the text collection into a numerical sequence of words.
        
        :param text_collection: A list of tokenized texts
        :return: A list of numerical sequences and the total number of unique words
        """
        numerical_sequences = []
        for doc in text_collection:
            sequence = []
            for word in doc:
                if word not in self.mapping:  # Map word to index if not already mapped
                    self.mapping[word] = self.n_components
                    self.n_components += 1
                sequence.append(self.mapping[word])  # Append the index
            numerical_sequences.append(sequence)
        return numerical_sequences, self.n_components
