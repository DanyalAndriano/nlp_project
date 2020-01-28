from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from collections import Counter
import re

contractions = {"ain't": 'are not', 'aint': 'are not', "aren't": 'are not', 'arent': 'are not', 'cant': 'cannot', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', 'cause': 'because', "could've": 'could have', 'couldve': 'could have', "couldn't": 'could not', 'couldnt': 'could not', "couldn't've": 'could not have', "didn't": 'did not', 'didnt': 'did not', "doesn't": 'does not', 'doesnt': 'does not', "don't": 'do not', 'dont': 'do not', "hadn't": 'had not', 'hadnt': 'had not', "hadn't've": 'had not have', "hasn't": 'has not', 'hasnt': 'has not', "haven't": 'have not', 'havent': 'have not', "he'd": 'he would', 'hed': 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he will have', "he's": 'he is', 'hes': 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', 'hows': 'how is', "i'd": 'I would', 'id': 'I would', "i'd've": 'I would have', "i'll": 'I will', "i'll've": 'I will have', "i'm": 'I am', "I'm":'I am', 'im': 'I am', "i've": 'I have', 'ive': 'I have', "isn't": 'is not', 'isnt': 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', 'its': 'it is', "let's": 'let us', 'lets': 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', 'mightve': 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', 'mustve': 'must have', "mustn't": 'must not', 'mustnt': 'must not', "mustn't've": 'must not have', "needn't": 'need not', 'neednt': 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', 'shes': 'she is', "should've": 'should have', 'shouldve': 'should have', "shouldn't": 'should not', 'shouldnt': 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so is', "that'd": 'that had', "that'd've": 'that would have', "that's": 'that is', 'thats': 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', 'theres': 'there is', "they'd": 'they would', 'theyd': 'they would', "they'd've": 'they would have', "they'll": 'they will', 'theyll': 'they will', "they'll've": 'they will have', "they're": 'they are', 'theyre': 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', 'wasnt': 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', 'weve': 'we have', "weren't": 'were not', 'werent': 'were not', "what'll": 'what will', 'whatll': 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', 'whats': 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', 'whered': 'where did', "where's": 'where is', 'wheres': 'where is', "where've": 'where have', 'whereve': 'where have', "who'll": 'who will', 'wholl': 'who will', "who'll've": 'who will have', "who's": 'who is', 'whos': 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', 'wont': 'will not', "win't": 'will not', "won't've": 'will not have', "would've": 'would have', 'wouldve': 'would have', "wouldn't": 'would not', 'wouldnt': 'would not', "wouldn't've": 'would not have', "y'all": 'you all', 'yall': 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', 'youd': 'you would', "you'd've": 'you would have', "you'll": 'you will', 'youll': 'you will', "you'll've": 'you will have', "you're": 'you are', 'youre': 'you are', "you've": 'you have', 'youve': 'you have'}

def expand_contractions(str_input):
    """Expands contracted words by removing the apostrophe and 
    replacing with full words."""
    str_input2 = re.sub(r"[^A-Za-z0-9' ]+", " ", str_input).lower()
    new = re.sub(r"\s+\'", "'", str_input2)
    for word in new.split():
        if word in contractions.keys():
            new = str_input.replace(word, contractions[word])
    return new

def clean_text(str_input):
    """Returns strings with urls, hashtags, contractions, new line characters,
    and punctuation removed """
    no_url = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?'," ",str_input)
    no_hash = re.sub(r'(?:(?<=\s)|^)#(\w*[A-Za-z_-]+\w*)', ' ', no_url)
    no_contractions = expand_contractions(no_hash)
    no_newline = re.sub(r"(?<=[a-z])[\r\n]+"," ", no_contractions)
    no_punc = re.sub(r"[^A-Za-z0-9]+", " ", no_newline)
    return no_punc
    
    
def SelectText(X):
    return X['text']

def SelectSentiment(X):
    return X[['rating', 'aws_neg', 'aws_pos', 'aws_mixed']]

def TextFeatures(X):
    text_features = []
    for i in X.index:
        str_input = clean_text(X[i]).lower()
        word_count = len(str_input.split())
        unique_words = len(set(str_input.split()))
        char_count = len(str_input)
        hashtags = len([x for x in X[i].split() if x.startswith('#')])
        happy_face = len([c for c in X[i].split() if c in [':)', ':-)', ':D']])
        sad_face = len([c for c in X[i].split() if c in [':(', ':-(', ':((']])
        exclaim = Counter(c for text in X[i] for c in text if c in ["!"])['!']
        question = Counter(c for text in X[i] for c in text if c in ["?"])['?']
        digits = len([x for x in str_input.split() if x.isdigit()])
        text_features.append([word_count, unique_words, char_count,
                                  hashtags, happy_face, sad_face, exclaim, question,
                                  digits])
    return text_features

