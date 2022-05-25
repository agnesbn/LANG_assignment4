"""
Classification using standard machine learning
"""
""" Import relevant packages """
 # system tools
import os
import sys
 # argument parser
import argparse
 # data wranling tools
import pandas as pd
sys.path.append(os.path.join("."))
import utils.classifier_utils as clf
 # machine learning tools
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
 # visualisation
import matplotlib.pyplot as plt
import seaborn as sns
 # surpress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=DeprecationWarning)

""" Basic functions """
# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", 
                    "--report_name", 
                    default="machine_classification_report", 
                    help="The name you wish to save the classification report under")
    args = vars(ap.parse_args())
    return args

# Function for saving the classification report
def report_to_txt(report, report_name):
    outpath = os.path.join("out", "machine",f"{report_name}.txt")
    with open(outpath,"w") as file:
        file.write(str(report))
        
""" Classification function """        
def classification():
    # load and read the data
    filename = os.path.join("in", "toxic", "VideoCommentsThreatCorpus.csv")
    data = pd.read_csv(filename)
    # change 0's and 1's to labels, "non-toxic" and "toxic"
    data["label"].replace({0:"non-toxic", 1:"toxic"}, inplace = True)
    # create balanced data
    data_balanced = clf.balance(data, 1000)
    X = data_balanced["text"]
    y = data_balanced["label"]
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # vectorisation and feature extraction
    vectorizer = CountVectorizer(ngram_range = (1,2), 
                                 lowercase = True, 
                                 max_df = 0.95, 
                                 min_df = 0.05, 
                                 max_features = 500)
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names()
    # classifying
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats, y_train)
    # predicting
    y_pred = classifier.predict(X_test_feats)
    # evaluate
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    return classifier_metrics

""" Main function """
def main():
    # parse arguments
    args = parse_args()
    # get parameters
    report_name = args["report_name"]
    # do classification
    report = classification()
    # save report
    report_to_txt(report, report_name)
    # print report
    return print(report)

if __name__ == "__main__":
    main()

