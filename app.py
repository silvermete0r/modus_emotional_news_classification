import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
from collections import defaultdict

st.set_page_config(
    page_title="Modus",
    page_icon=":newspaper:",
    layout="wide",
    initial_sidebar_state="expanded"
)

get_emoji = defaultdict(lambda: "😐", {
    "happy": "😊",
    "good": "🙂",
    "normal": "😐",
    "sad": "😔",
    "angry": "😡"
})

@st.cache_resource()
def load_sentiment_model():
    sentiment_analysis = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment",top_k=3)
    return sentiment_analysis

sentiment_analysis = load_sentiment_model()

def classify_emotion(text):
    sentiment_results = sentiment_analysis(text)[0]
    ans_score, ans_mood = 0, ""
    for sent in sentiment_results:
        if sent['score'] > ans_score:
            ans_score, ans_mood = sent['score'], sent['label']
    if ans_mood == 'LABEL_2':
        if ans_score < 0.9:
            ans_mood = 'good'
        else:
            ans_mood = 'happy'
    elif ans_mood == 'LABEL_1':
        ans_mood = 'normal'
    else:
        if ans_score < 0.9:
            ans_mood = 'sad'
        else:
            ans_mood = 'angry'
            
    return ans_mood, ans_score

@st.cache_data()
def get_news(query, country, category, api_key):
    if query == "Modus News":
        query = ""
    params = {
        'q': query,
        'country': country,
        'category': category,
        'pageSize': 100,
        'apiKey': api_key
    }
    base_url = 'https://newsapi.org/v2/top-headlines'
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f'Error: {e}')
        return None

    articles = {
        'title': [],
        'content': [],
        'url': [],
        'imgUrl': [],
    }

    for article in data['articles']:
        articles['title'].append(article.get('title', ''))
        articles['content'].append(article.get('description', ''))
        articles['url'].append(article.get('url', ''))
        articles['imgUrl'].append(article.get('urlToImage', ''))

    news_df = pd.DataFrame(articles)

    if len(news_df) == 0:
        st.warning('No articles found.')
        return None

    news_df['emotion'], news_df['score'] = zip(*news_df['title'].map(classify_emotion))

    return news_df

st.title("Modus")
st.markdown("## Your Personal News Assistant")

with st.sidebar:
    st.image('media/logo.png', use_column_width=True, output_format='PNG')
    st.title("Settings")
    
    st.subheader("Search")
    query = st.text_input("Search", "Modus News")
    
    st.subheader("Country")
    country = st.selectbox("Country", ["us", "gb", "au"], index=0)

    st.sidebar.subheader("Category")
    category = st.sidebar.selectbox("Category", ["business", "entertainment", "general", "health", "science", "sports", "technology"], index=0)

    st.sidebar.subheader("Mood")
    mood = st.sidebar.selectbox("Mood", ["happy", "good", "normal", "sad", "angry"], index=0)

    st.info('This is a news assistant that helps you to get relevant news classified by emotions. It uses the latest NLP and ML techniques to classify the news articles into different emotions.')
    st.caption('Made with ❤️ by SkillDeti team.')

news_df = get_news(query, country, category, st.secrets["NEWS_API_KEY"])

news_df = news_df[news_df['emotion'] == mood]

if news_df is not None:
    st.write('Total Articles:', len(news_df))

    if news_df is not None:
        for index, row in news_df.iterrows():
            st.write('---')
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(row['imgUrl'] if row['imgUrl'] else 'media/logo.png', use_column_width=True)
            with col2:
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Content:** {row['content']}")
                st.write(f"**URL:** {row['url']}")
                st.write(f"**Emotion:** {get_emoji[row['emotion']]} {row['emotion']} ({row['score']})")
else:
    st.warning('No articles found.')