# Assignment 3 - Text classification
The portfolio for __Language Analytics S22__ consists of 5 projects (4 class assignments and 1 self-assigned project). This is the __fourth assignment__ in the portfolio. 

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in.
Utils made by Ross.

## 2. Assignment description by Ross
### Main task
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

### Bonus task
- Add a range of different ```Argparse``` parameters that would allow the user to interact with the code, such as the embedding dimension size, the CountVector parameters.
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
# required packages
pip install pandas numpy scipy spacy tqdm spacytextblob vaderSentiment networkx scikit-learn tensorflow gensim textsearch contractions nltk beautifulsoup4 transformers autocorrect pytesseract opencv-python
# install spacy model
python -m spacy download en_core_web_sm
# ocr tools
sudo apt install -y tesseract-ocr
sudo apt install -y libtesseract-dev
```

### Get the data

### Main task


### Bonus task


## 5. Discussion of results
