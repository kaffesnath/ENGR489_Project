import re
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
text = "@xderbusfahrer Thanks for entering Grand Summoners  . . Watch the video to see if you won a $100 Amazon gift card! . Retweet every day for another chance to win!. . . Play GS Global now for a FREE . 5 One Punch Man Unit!.  https://t.co/8uSIKCQEO2"
wnl = WordNetLemmatizer()

#remove links
print("-----------------------------------Original-----------------------------------")
print(text)
print("\n")
text = str(text)
text = re.sub('(https?:\/\/)?(\w+\.)?(\w+\.)?\w+\.\w+(\s*\/\s*\w+)*', '', text) 
print("-------------------------------Removal of Links-------------------------------")
print(text)
print("\n")
#remove mentions or user handles
text = re.sub('(^|\B)@ \w+', '', text)
text = re.sub('(^|\B)@\w+', '', text)
print("-------------------------------Removal of Tags--------------------------------")
print(text)
print("\n")
text = re.sub('(<(.*?)>)', '', text)
print("-------------------------------Removal of Emojis------------------------------")
print(text)
print("\n")
#remove special characters
text = re.sub(r'[^a-zA-Z ]', '', text)
print("------------------------Removal of Special Characters-------------------------")
print(text)
print("\n")
text = text.lower()
text = re.sub(r' +', ' ', text).strip()
print("--------------------------Cleaning of remaining Text--------------------------")
print(text)
print("\n")
#removes stopwords and stems the words
print("-----------------------------Removal of Stopwords-----------------------------")
text = remove_stopwords(text)
print(text)
print("\n")
#text = PorterStemmer().stem(text)
text = ' '.join([wnl.lemmatize(word) for word in text.split()])
print("----------------------------Lemmatization of words----------------------------")
print(text)
print("\n")