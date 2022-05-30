# Assignment 4 - Text classification
The portfolio for __Language Analytics S22__ consists of 5 projects (4 class assignments and 1 self-assigned project). This is the __fourth assignment__ in the portfolio. 

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in.

The [classifier_utils.py](https://github.com/agnesbn/LANG_assignment4/blob/main/utils/classifier_utils.py) script in the `utils` folder was made by Ross and used in the [machine learning script](https://github.com/agnesbn/LANG_assignment4/blob/main/src/classification_machine.py). Furthermore, the [save_history()](https://github.com/agnesbn/LANG_assignment4/blob/e7665a2f5aebc731ec1c387a72eb6ddaf1469cc1/src/classification_deep.py#L94) function was inspired by one provided by Ross during the course.

## 2. Assignment description by Ross
### Main task
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using __standard machine learning approaches__
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of __deep learning methods__ we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

### Bonus task
- Add a range of __different ```Argparse``` parameters__ that would allow the user to interact with the code, such as the __embedding dimension size, the CountVector parameters__.
  - Think about which parameters are most *likely* to be modified by a user.

## 3. Methods
### Main task


### Bonus task



## 4. Usage
### Install packages
Before running the script, you have to install the relevant packages. To do this, run the following from the command line:
```
sudo apt update
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow contractions nltk
```

### Get the data

### Main task


### Bonus task


## 5. Discussion of results
