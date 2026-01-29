import sys

import pandas as pd
import pythainlp
import nltk
from nltk.tokenize import word_tokenize
import math


class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)

    def compute_probability(self, text_string):
        # words = pythainlp.tokenize.word_tokenize(text_string,keep_whitespace = False)

        tokens = set(word_tokenize(text_string))
        labels = self.get_all_possible_labels()
        features = self.get_all_possible_features()

        filtered_tokens = tokens.intersection(features)

        # Dict (label -> score)
        score_dict = {}

        for label in labels:
            score_dict[label] = 0
            for token in filtered_tokens:
                score_dict[label] += self.model_params.loc[token, label]

        # Calculate probability exp(A_i) / sum(exp(A_j))
        prob_dict = {}
        for label in labels:
            prob_dict[label] = math.exp(score_dict[label]) / sum(math.exp(score_dict[label]) for label in labels)
        
        return prob_dict

    def get_all_possible_features(self):
        # out -> List[str]
        return self.model_params.index.tolist()

    def get_all_possible_labels(self):
        # out -> List[str]
        return self.model_params.columns.tolist()

    def classify(self, text_string):
        pass


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tpython logistic_regression.py <model_file>')
        sys.exit(0)
    # argv[1] is csv file for model params
    model_file_name = sys.argv[1]
    model = TextClassifier(model_file_name)
    print('#---------------#')
    print("Feature :",model.get_all_possible_features())
    print("Label :",model.get_all_possible_labels())
    print('#---------------#')
    print(model.compute_probability("I like money and hate dust"))

