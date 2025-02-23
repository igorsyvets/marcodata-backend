from multiprocessing import process
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sentence_transformers import SentenceTransformer
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sentence_transformers import SentenceTransformer
from typing import List
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from mistralai import Mistral
import json



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
    num_topics: int = 15  # Higher = more granular topics, but potentially less coherent
    min_cluster_size: int = 5  # Minimum posts per topic (smaller = more topics but less reliable)
    max_words_per_topic: int = 2  # Number of words in topic name (3-5 recommended)
    similarity_threshold: float = 0.5  # How similar posts must be to stay in same cluster (0.5-0.8 typical)
    max_iterations: int = 300  # Maximum clustering iterations (higher = better clusters but slower)
    custom_stopwords: List[str] = ['apple']  # Words to ignore in both clustering and naming
    min_topic_coherence: float = 0.8  # Required topic similarity (0.7-0.9 recommended)
    outlier_threshold: float = 0.8  # When to move posts to "Other" (lower = stricter clustering)
    
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

def get_ngram_features(
    texts: List[str], 
    target_dim: int = 384, 
    custom_stopwords: List[str] = [],
    ngram_weights: Dict[str, float] = {"unigrams": 1, "bigrams": 1, "trigrams": 1},  # Weight control
) -> np.ndarray:
    """
    Generate weighted n-gram features with dimensionality reduction.

    Parameters:
    - texts: List of text samples.
    - target_dim: Feature vector size.
    - custom_stopwords: Additional stopwords.
    - ngram_weights: Dict specifying importance of unigrams, bigrams, trigrams.
    """
    all_stopwords = set(ENGLISH_STOP_WORDS).union(custom_stopwords)

    # Define n-gram vectorizers with independent pipelines
    unigram_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=target_dim // 3,
            stop_words=list(all_stopwords),
            sublinear_tf=True,
            min_df=5
        ))
    ])

    bigram_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            ngram_range=(2, 2),
            max_features=target_dim // 3,
            stop_words=list(all_stopwords),
            sublinear_tf=True,
            min_df=5
        ))
    ])

    trigram_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            ngram_range=(3, 3),
            max_features=target_dim // 3,
            stop_words=list(all_stopwords),
            sublinear_tf=True,
            min_df=2
        ))
    ])

    # Feature extraction pipeline
    vectorizer = FeatureUnion([
        ("unigrams", unigram_pipeline),
        ("bigrams", bigram_pipeline),
        ("trigrams", trigram_pipeline),
    ])

    # Extract raw TF-IDF features
    tfidf_features = vectorizer.fit_transform(texts)

    # Convert to NumPy array if needed
    tfidf_features = tfidf_features.toarray()

    # Apply individual n-gram importance weights
    num_features = tfidf_features.shape[1]
    num_sections = num_features // 3

    unigram_weight = ngram_weights.get("unigrams", 1.0)
    bigram_weight = ngram_weights.get("bigrams", 1.5)
    trigram_weight = ngram_weights.get("trigrams", 2.0)

    tfidf_features[:, :num_sections] *= unigram_weight
    tfidf_features[:, num_sections:2*num_sections] *= bigram_weight
    tfidf_features[:, 2*num_sections:] *= trigram_weight

    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=min(target_dim, tfidf_features.shape[1] - 1))
    reduced_features = svd.fit_transform(tfidf_features)

    # Normalize final feature vector
    norms = np.linalg.norm(reduced_features, axis=1, keepdims=True) + 1e-8
    normalized_features = reduced_features / norms

    # Ensure the n-gram features have the same shape as the target dimension
    if normalized_features.shape[1] < target_dim:
        padding = np.zeros((normalized_features.shape[0], target_dim - normalized_features.shape[1]))
        normalized_features = np.hstack((normalized_features, padding))
    elif normalized_features.shape[1] > target_dim:
        normalized_features = normalized_features[:, :target_dim]

    return normalized_features[:, :target_dim]  # Trim if needed

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




def sent_mistral_prompt(prompt: str):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    if chat_response.choices:
        print(chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content
    else:
        print("No response from Mistral API")
        return ""

def generate_topic_names(clusters: Dict[str, Dict[str, List[str]]], max_words: int = 3) -> Dict[str, str]:
    """
    Generate topic names for all clusters using a single Mistral request.
    """
    # Prepare the combined request payload
    request_payload = []
    for topic_id, cluster in clusters.items():
        combined_text = " ".join(cluster['texts'][:10])
        request_payload.append({"topic_id": str(topic_id), "combined_text": combined_text})
    
    # Send request to Mistral
    response = sent_mistral_prompt(f"""
        You receive an array of objects. Each object contains a topic_id and a combined_text.
        Your task is to generate a name for each topic based on the combined_text.
        Please use the following guidelines:
        1. The name should be descriptive and concise.
        2. The name should be 3-5 words long.
        3. The name should be relevant to the combined_text.
        4. When choosing summary words, consider the most frequent and important terms such as
           brand names, unique words, person names, or geographical locations.
        5. Try your best to identify one key term that represents the topic
        6. Words should have logical order
        Here is the request payload:
        <request_payload>
        {request_payload}
        </request_payload>

        Response format should be in JSON. Example:
        <response_example>
        [
            {{\"topic_id\": \"1\", \"topic_name\": \"Apple iPhone 13\"}},
            {{\"topic_id\": \"2\", \"topic_name\": \"Tesla Electric Vehicles\"}},
            {{\"topic_id\": \"3\", \"topic_name\": \"SpaceX Mars Missions\"}}
        ]
        </response_example>
        
        Please DON'T include anything in the response other than a valid JSON code.
        Follow the response example format precisely. 
        Do not include backticks.
    """)

    print(f"\n\nMistral response: {response}\n\n")

    # Parse the response
    topic_names = {}

    ## add fallback names
    for topic_id in clusters.keys():
        topic_names[str(topic_id)] = f"Topic {topic_id}"

    if response:
        try:
            response_data = json.loads(response)  # Parse JSON response
            if isinstance(response_data, list):
                for item in response_data:
                    if isinstance(item, dict):
                        topic_id = item.get("topic_id")
                        topic_name = item.get("topic_name")
                        if topic_id and topic_name:
                            topic_names[str(topic_id)] = topic_name
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
    
    return topic_names

@app.post("/analyze", response_model=List[TopicResponse])
async def analyze_posts(request: PostsRequest):
    """
    Clusters posts into topics with configurable parameters.
    
    Key Parameters:
    --------------
    num_topics: int = 15
        - Controls maximum number of potential topics
        - Higher values create more specific but possibly noisier topics
        - Lower values create broader, more general topics
        - Actual number may be less based on data coherence
    
    min_cluster_size: int = 5
        - Minimum number of posts required to form a topic
        - Too low: Many small, potentially noisy topics
        - Too high: Forces posts into "Other" category
        - Should be adjusted based on total post count
    
    similarity_threshold: float = 0.6
        - How similar posts must be to stay in same cluster
        - 0.5: More permissive clustering, larger topics
        - 0.7: Stricter clustering, more coherent but fewer topics
        - 0.8+: Very strict, might move most posts to "Other"
    
    max_words_per_topic: int = 3
        - Number of words used in topic names
        - More words = more descriptive but potentially noisier
        - Recommended range: 2-5 words
    
    custom_stopwords: List[str] = []
        - Words to completely ignore in both clustering and naming
        - Example: ["apple", "iphone"] for Apple-related posts
        - Applied to both initial clustering and topic naming
        - Case insensitive
    
    min_topic_coherence: float = 0.8
        - Required semantic similarity within topics
        - Higher values create more focused topics
        - Lower values allow more diverse content per topic
    
    outlier_threshold: float = 0.5
        - Threshold for moving posts to "Other"
        - Lower values create stricter clustering
        - Higher values keep more posts in main topics
    
    Clustering Process:
    -----------------
    1. First pass: Uses strict parameters for main topics
    2. Second pass: Uses relaxed parameters for remaining posts
    3. Posts not fitting any cluster go to "Other"
    
    Returns:
    --------
    List[TopicResponse]: List of topics with their posts and names
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
    ngram_embeddings = get_ngram_features(
        texts, 
        custom_stopwords=request.custom_stopwords
    )
    
    # Combine embeddings
    embeddings = combine_embeddings(
        semantic_embeddings, 
        ngram_embeddings,
        alpha=1  # Adjust this to control importance of word order
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
            merged_clusters[label] = cluster
            categorized_posts.update(cluster['ids'])

    # Second pass: Try to cluster remaining posts with lower threshold
    remaining_indices = [i for i, pid in enumerate(post_ids) if pid not in categorized_posts]
    if remaining_indices:
        remaining_count = len(remaining_indices)
        print(f"Second pass clustering of {remaining_count} uncategorized posts...")
        
        # Adjust parameters for second pass
        adjusted_min_size = max(3, request.min_cluster_size // 2)  # More aggressive size reduction
        relaxed_threshold = request.similarity_threshold * 0.7  # More aggressive threshold reduction
        
        remaining_embeddings = embeddings[remaining_indices]
        remaining_texts = [texts[i] for i in remaining_indices]
        remaining_ids = [post_ids[i] for i in remaining_indices]
        
        secondary_clusters = try_cluster_posts(
            post_ids=remaining_ids,
            texts=remaining_texts,
            embeddings=remaining_embeddings,
            similarity_threshold=relaxed_threshold,  # Use relaxed threshold
            min_cluster_size=adjusted_min_size,  # Use adjusted minimum
            max_iterations=request.max_iterations  # Pass max_iterations
        )

        # Process secondary clusters with same size requirements
        for label, cluster in secondary_clusters.items():
            if len(cluster['texts']) > 0:
                if label == 'other' or len(cluster['texts']) < adjusted_min_size:
                    # Move to global 'other' topic
                    merged_clusters["other"]['texts'].extend(cluster['texts'])
                    merged_clusters['other']['ids'].extend(cluster['ids'])
                    merged_clusters['other']['display_name'] = "Other"  # Simple name for other cluster
                else:
                    merged_clusters[label] = cluster

    print(f"Final distribution: {sum(len(c['ids']) for c in merged_clusters.values())} posts in {len(merged_clusters)} topics")

    # Generate topic names for all clusters
    topic_names = generate_topic_names(merged_clusters, max_words=request.max_words_per_topic)

    # Assign topic names to clusters
    for canonical_key, cluster in merged_clusters.items():
        if canonical_key in topic_names:
            cluster['display_name'] = topic_names[canonical_key]

    # Generate response
    response = [
        TopicResponse(
            name=cluster.get('display_name', '') or topic_names.get(str(canonical_key), str(canonical_key)),  # Ensure canonical_key is a string
            id=f"topic_{i+1}",
            postIds=cluster['ids'],
            canonical_key=str(canonical_key)  # Ensure canonical_key is a string
        )
        for i, (canonical_key, cluster) in enumerate(merged_clusters.items())
    ]

    return response
