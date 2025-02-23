from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, field_validator
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
from collections import defaultdict
from numpy.typing import NDArray
import numpy.typing as npt
import scipy.sparse as sp
from sklearn.decomposition import PCA

# Create FastAPI app and router
app = FastAPI()
router = APIRouter(prefix="")  # Explicitly set empty prefix

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # Load environment variables from .env file

# Include router BEFORE any @app decorators
app.include_router(router)  # Remove the prefix to match original endpoint paths

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

# Helper function that cleans and normalizes input text
def preprocess_text(text: str, custom_stopwords: List[str] = []) -> str:
    """
    Preprocess text by removing URLs, mentions, and stop words
    Args:
        text: Input text to preprocess
        custom_stopwords: Additional words to filter out
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
    # Combine standard stop words with custom domain stopwords
    all_stopwords = ENGLISH_STOP_WORDS.union(set(custom_stopwords))
    # Remove stop words and lowercase
    words = [word.lower() for word in text.split() if word.lower() not in all_stopwords]
    return ' '.join(words) if words else ''

# Data models for request/response validation
class Post(BaseModel):
    id: str
    text: str

class PostsRequest(BaseModel):
    posts: List[Post]
    num_topics: int = 15  # Default number of topics to generate
    min_cluster_size: int = 5  # Minimum number of posts per topic
    max_words_per_topic: int = 3  # Number of words to use in topic name
    similarity_threshold: float = 0.6  # Threshold for considering posts similar (0.0 to 1.0)
    max_iterations: int = 300  # Maximum iterations for K-means clustering
    custom_stopwords: List[str] = []  # Domain-specific words to ignore (e.g., ["apple", "twitter"])
    min_topic_coherence: float = 0.5  # Minimum semantic similarity within topic
    outlier_threshold: float = 0.3    # Threshold for considering a post as outlier
    
    # Validate parameters
    @field_validator('num_topics')
    def validate_num_topics(cls, v):
        if v < 1:
            raise ValueError('num_topics must be at least 1')
        return v

    @field_validator('min_cluster_size')
    def validate_min_cluster_size(cls, v):
        if v < 1:
            raise ValueError('min_cluster_size must be at least 1')
        return v

    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('similarity_threshold must be between 0.0 and 1.0')
        return v

    @field_validator('custom_stopwords')
    def normalize_stopwords(cls, v):
        # Convert stopwords to lowercase for consistent matching
        return [word.lower().strip() for word in v]

class TopicResponse(BaseModel):
    name: str        # Display name with meaningful word order
    id: str         # Unique identifier for the topic
    postIds: List[str]  # List of post IDs belonging to this topic
    canonical_key: str  # Hidden field for deduplication

def get_canonical_key(words: List[str]) -> str:
    """
    Creates a canonical key for deduplication by sorting words alphabetically
    """
    return " ".join(sorted(set(word.lower() for word in words)))

def normalize_topic_name(words: List[str], scores: List[float]) -> tuple[str, str]:
    """
    Creates both a display name and a canonical key for a topic
    
    Args:
        words: List of words in topic
        scores: TF-IDF scores for words
    Returns:
        Tuple of (display_name, canonical_key)
    """
    # Sort by TF-IDF scores for display name
    word_scores = list(zip(words, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates while preserving score-based order
    seen = set()
    display_words = []
    for word, _ in word_scores:
        word_lower = word.lower()
        if word_lower not in seen:
            seen.add(word_lower)
            display_words.append(word_lower)
    
    display_name = " ".join(display_words)
    canonical_key = get_canonical_key(words)
    
    return display_name, canonical_key

def calculate_semantic_coherence(texts: List[str], embeddings: NDArray[np.float64]) -> float:
    """
    Calculate how semantically coherent a group of texts is
    Returns score between 0-1 where higher means more coherent
    """
    # Calculate pairwise similarities
    similarities: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = float(np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            ))
            similarities.append(similarity)
    
    return float(np.mean(similarities) if similarities else 0.0)

def create_cluster_dict():
    """Helper function to create properly initialized cluster dictionary"""
    return {
        'texts': [],
        'ids': [],
        'display_name': ''
    }

def try_cluster_posts(post_ids: List[str], texts: List[str], embeddings: np.ndarray, 
                     similarity_threshold: float, min_cluster_size: int,
                     max_iterations: int = 300) -> Dict:
    """Helper function to cluster posts with given parameters"""
    print(f"Attempting to cluster {len(texts)} posts (min size: {min_cluster_size})")
    
    if len(texts) < min_cluster_size:
        print(f"Not enough posts ({len(texts)}) for clustering, moving all to 'other'")
        return {"other": {'texts': texts, 'ids': post_ids}}
        
    # Adjust number of clusters to ensure minimum size
    max_possible_clusters = len(texts) // min_cluster_size
    n_clusters = min(max(2, max_possible_clusters), max_possible_clusters)
    print(f"Creating {n_clusters} clusters (max possible: {max_possible_clusters})")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=max_iterations
    )
    kmeans.fit(embeddings)
    
    # First, group by labels
    temp_clusters = defaultdict(lambda: {'texts': [], 'ids': [], 'embeddings': []})
    for idx, label in enumerate(kmeans.labels_):
        temp_clusters[label]['texts'].append(texts[idx])
        temp_clusters[label]['ids'].append(post_ids[idx])
        temp_clusters[label]['embeddings'].append(embeddings[idx])
    
    # Debug print cluster sizes
    for label, cluster in temp_clusters.items():
        print(f"Initial cluster {label}: {len(cluster['ids'])} posts")
    
    # Validate clusters and redistribute small ones
    clusters = defaultdict(lambda: {'texts': [], 'ids': []})
    for label, cluster in temp_clusters.items():
        if len(cluster['ids']) >= min_cluster_size:
            # For each post in valid-sized cluster, check similarity
            cluster_center = np.mean(cluster['embeddings'], axis=0)
            similar_posts = 0
            
            for idx, emb in enumerate(cluster['embeddings']):
                similarity = np.dot(emb, cluster_center) / (
                    np.linalg.norm(emb) * np.linalg.norm(cluster_center)
                )
                if similarity >= similarity_threshold:
                    similar_posts += 1
                    clusters[label]['texts'].append(cluster['texts'][idx])
                    clusters[label]['ids'].append(cluster['ids'][idx])
                else:
                    clusters['other']['texts'].append(cluster['texts'][idx])
                    clusters['other']['ids'].append(cluster['ids'][idx])
            
            # If after similarity check cluster becomes too small, move all to other
            if similar_posts < min_cluster_size:
                print(f"Cluster {label} too small after similarity check ({similar_posts} posts), moving to other")
                clusters['other']['texts'].extend(clusters[label]['texts'])
                clusters['other']['ids'].extend(clusters[label]['ids'])
                del clusters[label]
        else:
            print(f"Cluster {label} too small ({len(cluster['ids'])} posts), moving to other")
            clusters['other']['texts'].extend(cluster['texts'])
            clusters['other']['ids'].extend(cluster['ids'])
    
    # Final size check
    final_clusters = defaultdict(lambda: {'texts': [], 'ids': []})
    for label, cluster in clusters.items():
        if label != 'other' and len(cluster['ids']) >= min_cluster_size:
            final_clusters[label] = cluster
        else:
            final_clusters['other']['texts'].extend(cluster['texts'])
            final_clusters['other']['ids'].extend(cluster['ids'])
    
    print(f"Final clusters: {sum(1 for l in final_clusters if l != 'other')} valid clusters")
    for label, cluster in final_clusters.items():
        print(f"Final cluster {label}: {len(cluster['ids'])} posts")
    
    return final_clusters

def get_ngram_features(texts: List[str], target_dim: int = 384) -> np.ndarray:
    """
    Generate n-gram features to capture word order and normalize dimensions
    """
    # Use both unigrams and bigrams with limited features
    ngram_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=target_dim,  # Limit to target dimension directly
        stop_words='english'
    )
    ngram_features = ngram_vectorizer.fit_transform(texts)
    dense_features = np.asarray(ngram_features.todense())
    
    # If we have fewer features than target_dim, pad with zeros
    if dense_features.shape[1] < target_dim:
        padding = np.zeros((dense_features.shape[0], target_dim - dense_features.shape[1]))
        return np.hstack([dense_features, padding])
    return dense_features

def combine_embeddings(semantic_emb: np.ndarray, ngram_emb: np.ndarray, 
                      alpha: float = 0.7) -> np.ndarray:
    """
    Combine BERT embeddings with n-gram features
    """
    print(f"Semantic embeddings shape: {semantic_emb.shape}")
    print(f"N-gram embeddings shape: {ngram_emb.shape}")
    
    # Ensure both embeddings are normalized
    semantic_norm = semantic_emb / (np.linalg.norm(semantic_emb, axis=1, keepdims=True) + 1e-8)
    ngram_norm = ngram_emb / (np.linalg.norm(ngram_emb, axis=1, keepdims=True) + 1e-8)
    
    return alpha * semantic_norm + (1 - alpha) * ngram_norm

def generate_topic_name(texts: List[str], max_words: int = 5) -> tuple[List[str], List[float]]:
    """
    Generate topic name from cluster texts with improved word selection
    """
    vectorizer = TfidfVectorizer(
        max_features=max_words * 3,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrix to dense and sum across documents
    tfidf_sums = np.asarray(tfidf_matrix.todense()).sum(axis=0).flatten()
    
    # Get all candidate words and scores
    word_scores = [(str(feature_names[i]), float(tfidf_sums[i])) 
                  for i in range(len(feature_names))]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates while preserving order
    seen_words = set()
    unique_words = []
    unique_scores = []
    
    for word, score in word_scores:
        parts = word.split()
        if not any(part in seen_words for part in parts):
            unique_words.append(word)
            unique_scores.append(score)
            seen_words.update(parts)
            
            if len(unique_words) >= max_words:
                break
    
    return unique_words, unique_scores

@app.post("/analyze", response_model=List[TopicResponse])
async def analyze_posts(request: PostsRequest):
    """
    Main endpoint for topic analysis of posts with configurable sensitivity:
    
    Parameters for tuning:
    - num_topics: More topics = finer granularity but possibly smaller clusters
    - min_cluster_size: Higher value = fewer but more coherent topics
    - max_words_per_topic: More words = more descriptive but possibly noisier topics
    - similarity_threshold: Higher value = stricter similarity requirements
    - max_iterations: More iterations = potentially better clustering but slower
    
    Custom stopwords:
    - Add domain-specific terms that should be ignored during clustering
    - Example: ["apple", "iphone"] for Apple-related posts
    - These words won't be used in topic names or similarity calculations
    """
    # Input validation
    if not request.posts:
        raise HTTPException(status_code=400, detail="No posts provided")
    
    # Preprocess all posts with custom stopwords
    cleaned_posts = [(post.id, preprocess_text(post.text, request.custom_stopwords)) for post in request.posts]
    post_ids, texts = zip(*cleaned_posts)
    
    # Convert tuples to lists after zip
    post_ids = list(post_ids)
    texts = list(texts)
    
    # Generate both semantic and n-gram embeddings
    semantic_embeddings = model.encode(texts)
    ngram_embeddings = get_ngram_features(texts)
    
    # Combine embeddings
    embeddings = combine_embeddings(
        semantic_embeddings, 
        ngram_embeddings,
        alpha=0.7  # Adjust this to control importance of word order
    )

    # First pass: Try with original parameters
    print(f"Initial clustering of {len(post_ids)} posts...")
    main_clusters = try_cluster_posts(
        post_ids=post_ids,  # Add explicit parameter names
        texts=texts,
        embeddings=embeddings,
        similarity_threshold=request.similarity_threshold,
        min_cluster_size=request.min_cluster_size,
        max_iterations=request.max_iterations  # Pass max_iterations
    )

    # Track categorized posts
    categorized_posts = set()
    merged_clusters = defaultdict(create_cluster_dict)

    # Process main clusters
    for label, cluster in main_clusters.items():
        if (label != 'other' and 
            len(cluster['texts']) >= request.min_cluster_size):  # Strict size check
            # Process cluster and generate topic
            top_words, top_scores = generate_topic_name(
                cluster['texts'], 
                max_words=request.max_words_per_topic
            )
            
            display_name, canonical_key = normalize_topic_name(top_words, top_scores)
            
            merged_clusters[canonical_key]['texts'].extend(cluster['texts'])
            merged_clusters[canonical_key]['ids'].extend(cluster['ids'])
            merged_clusters[canonical_key]['display_name'] = display_name
            
            categorized_posts.update(cluster['ids'])

    # Second pass: Try to cluster remaining posts with lower threshold
    remaining_indices = [i for i, pid in enumerate(post_ids) if pid not in categorized_posts]
    if remaining_indices:
        remaining_count = len(remaining_indices)
        print(f"Second pass clustering of {remaining_count} uncategorized posts...")
        
        # Adjust minimum cluster size based on remaining posts
        adjusted_min_size = max(
            request.min_cluster_size,  # Never go below original minimum
            remaining_count // 10  # Try to create at most 10 secondary clusters
        )
        
        remaining_embeddings = embeddings[remaining_indices]
        remaining_texts = [texts[i] for i in remaining_indices]
        remaining_ids = [post_ids[i] for i in remaining_indices]
        
        secondary_clusters = try_cluster_posts(
            post_ids=remaining_ids,
            texts=remaining_texts,
            embeddings=remaining_embeddings,
            similarity_threshold=request.similarity_threshold * 0.8,  # Slightly relaxed
            min_cluster_size=adjusted_min_size,  # Use adjusted minimum
            max_iterations=request.max_iterations  # Pass max_iterations
        )

        # Process secondary clusters with same size requirements
        for label, cluster in secondary_clusters.items():
            if len(cluster['texts']) > 0:
                if label == 'other' or len(cluster['texts']) < adjusted_min_size:
                    # Move to global 'other' topic
                    merged_clusters["other"]['texts'].extend(cluster['texts'])
                    merged_clusters["other"]['ids'].extend(cluster['ids'])
                    if label == 'other':
                        topic_name = "other"
                        if cluster['texts']:
                            # Generate topic name for 'other' cluster
                            top_words, _ = generate_topic_name(cluster['texts'], max_words=3)
                            topic_name = "other: " + " ".join(top_words)
                        merged_clusters["other"]['display_name'] = topic_name
                else:
                    # Create secondary topic
                    top_words, top_scores = generate_topic_name(
                        cluster['texts'], 
                        max_words=request.max_words_per_topic
                    )
                    display_name, canonical_key = normalize_topic_name(top_words, top_scores)
                    canonical_key = f"secondary_{canonical_key}"
                    
                    merged_clusters[canonical_key]['texts'].extend(cluster['texts'])
                    merged_clusters[canonical_key]['ids'].extend(cluster['ids'])
                    merged_clusters[canonical_key]['display_name'] = display_name

    print(f"Final distribution: {sum(len(c['ids']) for c in merged_clusters.values())} posts in {len(merged_clusters)} topics")

    # Generate response
    response = [
        TopicResponse(
            name=cluster['display_name'] or canonical_key,  # Fallback to canonical key if no display name
            id=f"topic_{i+1}",
            postIds=cluster['ids'],
            canonical_key=canonical_key
        )
        for i, (canonical_key, cluster) in enumerate(merged_clusters.items())
    ]

    return response