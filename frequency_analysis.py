from collections import Counter

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from data_loader import getDataframe

if __name__ == "__main__":
    # nltk.download('stopwords')

    stopWords = set(stopwords.words("english"))
    df = getDataframe('data_100.csv')

    def is_not_stop_word(word: str) -> bool:
        if len(word) == 1:
            return False
        return word not in stopWords

    def does_not_have_special_chars(word: str) -> bool:
        return bool(re.search('^[a-z0-9]*$', word))

    def filter_stop_words(summary: str) -> list:
        all_words_lower_case = word_tokenize(summary.lower())
        words = list(filter(lambda s: is_not_stop_word(s) and does_not_have_special_chars(s), all_words_lower_case))
        return words

    def get_word_count(words: list) -> dict:
        return Counter(words)

    df['Words'] = df['Summary'].apply(filter_stop_words)
    df['Word Count'] = df['Words'].apply(get_word_count)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df['Word Count'])
