import sys

import pandas as pd
import pythainlp

class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)

    def compute_probability(self, text_string):
        pass

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

