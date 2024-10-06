# Spam Detection in SMS

## Project Overview

This project focuses on detecting spam messages in SMS using various natural language processing (NLP) techniques and machine learning models. The aim is to identify and classify text messages as either spam or non-spam (ham).

In this project, I explored several important NLP techniques such as data cleaning, text preprocessing, and vectorization methods. Finally, I built and compared machine learning models to evaluate their performance.

## Steps Involved

### 1. Data Cleaning and Preprocessing
- **Removing Punctuation:** To reduce noise in the text data.
- **Stop Words Removal:** Removed common words (like "is", "and", "the") that don't contribute much to the meaning.
- **Stemming and Lemmatization:** Converted words to their root forms for better consistency in text data.

### 2. Text Vectorization Techniques
I experimented with various vectorization methods to convert text data into numerical format:
- **Count Vectorization:** Converts text into a matrix of token counts.
- **N-grams:** Captures sequences of words, adding more context by considering word combinations.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their importance in the document, highlighting words that are more unique to specific messages.

### 3. Machine Learning Models
To classify SMS messages as spam or non-spam, I implemented and compared two models:
- **Random Forest Classifier**
- **XGBoost Classifier**

### 4. Model Evaluation
After building the models, I evaluated their performance using metrics like accuracy, precision, recall, and F1-score to determine which model performed best.

## Results
The results of the two models were compared, and their respective performance in terms of accuracy and other metrics were discussed to identify the most effective model for detecting spam.

## Technologies Used
- Python
- Pandas and NumPy for data manipulation
- NLTK for text preprocessing (tokenization, stop words, stemming, lemmatization)
- Scikit-learn for vectorization techniques and Random Forest
- XGBoost for the second machine learning model
- Matplotlib and Seaborn for visualizing model performance
