from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from typing import Dict, List
from urllib.parse import quote
import numpy as np
import os
import re
import requests
from dotenv import load_dotenv

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # Load environment variables from .env file

router = APIRouter()

@router.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """
    Basic test endpoint
    """
    return {"message": "API is working"}

@router.get("/tweets")
async def get_items() -> List[Dict[str, str]]:
    try:
        # Query parameters
        context = '"47,131",10026364281'
        search_term = "apple"
        language = "lang:en"
        location = "place_country:US"
        max_results = "1" # 10 is minimum
        
        # Combine parameters into base query
        base_query = f"{search_term} {language} {location} -is:retweet"
        encoded_query = quote(base_query)
        
        url = f"https://api.x.com/2/tweets/search/recent?query={encoded_query}&max_results={max_results}"
        headers = {"Authorization": f"Bearer {os.getenv('TWITTER_BEARER_TOKEN')}"}

        print("Trying with api key",os.getenv('TWITTER_BEARER_TOKEN'))

        response = requests.request("GET", url, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Body: {response.text}")
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        error_message = f"API request failed: {str(e)}"
        print(f"Error: {error_message}")
        print(f"Response Status Code: {getattr(e.response, 'status_code', 'N/A')}")
        print(f"Response Text: {getattr(e.response, 'text', 'N/A')}")
        
        raise HTTPException(
            status_code=500,
            detail=error_message
        )


# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing function
def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing URLs, mentions, and stop words
    Args:
        text: Input text to preprocess
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Keep hashtags but remove the '#' symbol
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    # Remove stop words and lowercase
    words = [word.lower() for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(words) if words else ''

# Request model
class Post(BaseModel):
    id: str
    text: str

class PostsRequest(BaseModel):
    posts: List[Post]
    num_topics: int = 15

# Response model
class TopicResponse(BaseModel):
    name: str
    id: str
    postIds: List[str]

@app.post("/analyze", response_model=List[TopicResponse])
async def analyze_posts(request: PostsRequest):
    if not request.posts:
        raise HTTPException(status_code=400, detail="No posts provided")

    # Create mapping of texts and IDs
    posts_map = {post.id: post.text for post in request.posts}
    
    # Preprocess posts while maintaining ID mapping
    cleaned_posts = [(post.id, preprocess_text(post.text)) for post in request.posts]
    post_ids, texts = zip(*cleaned_posts)
    
    # Generate embeddings
    embeddings = model.encode(list(texts))

    # Cluster using K-means
    kmeans = KMeans(n_clusters=request.num_topics, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Assign posts to clusters while keeping track of IDs
    clustered_posts = {i: {'texts': [], 'ids': []} for i in range(request.num_topics)}
    for idx, label in enumerate(labels):
        clustered_posts[label]['texts'].append(texts[idx])
        clustered_posts[label]['ids'].append(post_ids[idx])

    # Generate topic names using TF-IDF
    response = []
    for i in range(request.num_topics):
        if clustered_posts[i]['texts']:
            # Generate topic name
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(clustered_posts[i]['texts'])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_sums = np.asarray(tfidf_matrix.todense()).sum(axis=0)
            top_indices = tfidf_sums.argsort()[-3:][::-1]
            top_words = [str(feature_names[idx]) for idx in top_indices]
            topic_name = " ".join(top_words)
            
            # Create response object
            response.append(TopicResponse(
                name=topic_name,
                id=f"topic_{i+1}",
                postIds=clustered_posts[i]['ids']
            ))
        else:
            response.append(TopicResponse(
                name=f"Topic {i+1}",
                id=f"topic_{i+1}",
                postIds=[]
            ))

    return response