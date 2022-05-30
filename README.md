# Assignment 4 - Text classification
The portfolio for __Language Analytics S22__ consists of 5 projects (4 class assignments and 1 self-assigned project). This is the __fourth assignment__ in the portfolio. 

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in.

The [`classifier_utils.py`](https://github.com/agnesbn/LANG_assignment4/blob/main/utils/classifier_utils.py) script in the `utils` folder was made by Ross and used in the [machine learning script](https://github.com/agnesbn/LANG_assignment4/blob/5a222ed9800f66f328997483b57f09ab5d8c17f5/src/classification_machine.py#L13) and the [helper functions](https://github.com/agnesbn/LANG_assignment4/blob/17829f980283aa4cc606ae9ff04e47069a480122/src/classification_deep.py#L119), `strip_html_tags()`, `remove_accented_chars()`, and `pre_process_corpus()`, used in the [deep learning script](https://github.com/agnesbn/LANG_assignment4/blob/main/src/classification_deep.py) were also made by Ross. Furthermore, the [`save_history()`](https://github.com/agnesbn/LANG_assignment4/blob/e7665a2f5aebc731ec1c387a72eb6ddaf1469cc1/src/classification_deep.py#L94) function was inspired by one provided to us by Ross during the course.

## 2. Assignment description by Ross
### Main task
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using __standard machine learning approaches__
  - This means __```CountVectorizer()```__ or ```TfidfVectorizer()```, __```LogisticRegression``` classifier__
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of __deep learning methods__ we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

### Bonus task
- Add a range of __different ```Argparse``` parameters__ that would allow the user to interact with the code, such as the __embedding dimension size, the CountVector parameters__.
  - Think about which parameters are most *likely* to be modified by a user.

## 3. Methods
### Main task
#### Machine learning script
The [`classification_machine.py`](https://github.com/agnesbn/LANG_assignment4/blob/main/src/classification_machine.py) script first loads and reads a CSV, balances the set so that there are 1000 samples of each category, _toxic_ and _non-toxic_, and then splits the data into a training and a testing batch. Then, `CountVectorizer()` is used to convert the texts into a matrix of token counts, and these are used to train a logistic regression classifier. The model is used to predict labels for the testing data, and the results are saved in the form of a classification report.

#### Deep learning script
The [`classification_deep.py`](https://github.com/agnesbn/LANG_assignment4/blob/main/src/classification_deep.py) script similarly, loads and reads the data CSV, and performs a train/test split on the data. Using Ross' helper functions, the data is cleaned and normalised and any out-of-vocabulary tokens are replaced with `<UNK>`. The texts are turned into integer sequences and a max sequence length of 100 is defined. Zero-padding is added around the data matrix, so that if the kernel goes outside the matrix, the values will stay the same length. Now, it is possible to initialse a sequential model with an embedding layer, three convolutional layers, and a fully-connected classification layer. The model is compiled and trained on the data. Finally, the model is evaluated and the results are saved in the form of a classification report and a history plot.

### Bonus task
A number of `Argparse` parameters were added to the code. This allows for the user to change the name of the classification report for both scripts, and change the name of the history plot, the number of epochs, the embedding dimension size, and the batch size for the deep learning script. 

## 4. Usage
### Install packages
Before running the script, you have to install the relevant packages. To do this, run the following from the command line:
```
sudo apt update
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow contractions nltk
```

### Get the data
- The data belongs to the authors of [THREAT: A Large Annotated Corpus for Detection of Violent Threats](https://www.simula.no/sites/default/files/publications/files/cbmi2019_youtube_threat_corpus.pdf) and will be provided to the examiner by Ross.
- Place the data in the `in` folder so that the path to the data is `in/toxic/VideoCommentsThreatCorpus.csv`.


### Machine learning script

### Deep learning script


## 5. Discussion of results
