import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
first_time = True
if first_time:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
import html
import warnings
warnings.filterwarnings("ignore")


#QUESTIONS
pth_questions = 'data\Questions.csv'
df_questions = pd.read_csv(pth_questions, encoding='latin1')

#ANSWERS
pth_answers = 'data\Answers.csv'
df_answers = pd.read_csv(pth_answers, encoding='latin1')

#QUESTIONS + ANSWERS
merged_df = pd.merge(df_questions,df_answers, left_on='Id',  right_on='ParentId', how='inner')

#RENAME 
col_names = {
    'Id_x': 'question_id',
    'Body_x': 'question_text', 
    'Id_y': 'answer_id', 
    'Body_y': 'answer_text', 
}
merged_df = merged_df.rename(columns=col_names, inplace=False)
#merged_df = merged_df.drop(columns = [ 'OwnerUserId_x', 'CreationDate_x', 'ClosedDate', 'Score_x', 'ParentId', 'OwnerUserId_y', 'CreationDate_y',  'Score_y'])
merged_df = merged_df[['question_id', 'Title', 'question_text', 'answer_text']]


#DROP DUPLICATES
merged_df = merged_df.drop_duplicates(subset=['question_id'], keep='first')

#CLEAN DATA
def clean_html_thorough(text):
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "img", "iframe", "noscript"]):
            element.decompose()
    return soup.get_text(separator=" ")

def clean_html(text):
    text_html_removed = BeautifulSoup(text, "lxml").text
    text_html_removed = html.unescape(text_html_removed)             
    text_html_removed = clean_html_thorough(text_html_removed)       
    return text_html_removed

def tokenization(text):
    tokens = word_tokenize(text)
    tokens_cleaned = [word.lower() for word in tokens if word.lower() not in stop_words]
    return tokens_cleaned

merged_df['question_text'] = merged_df['question_text'].apply(clean_html)
merged_df['answer_text'] = merged_df['answer_text'].apply(clean_html)
merged_df['question_tokens'] = merged_df['question_text'].apply(tokenization)
merged_df['answer_tokens'] = merged_df['answer_text'].apply(tokenization)

#STRUCTURE DATA AS LST
data = merged_df.to_dict('records')