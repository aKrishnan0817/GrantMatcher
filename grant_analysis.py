import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
from collections import Counter
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer  # Or your preferred method
import os
# Initialize OpenAI client (You'll need to set the environment variable)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Example grant descriptions (REPLACE WITH YOUR DATA)
grant_descriptions = [
    # ... your grant descriptions
]

def get_openai_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def analyze_grants(grant_descriptions):
    # 1. Compute Embeddings
    embeddings = []
    for description in grant_descriptions:
        embedding = get_openai_embedding(description)
        embeddings.append(embedding)
    embeddings = np.array(embeddings).astype("float32")

    # 2. Determine Optimal k and Cluster
    optimal_k = find_optimal_clusters(embeddings)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    clusters = {i: [] for i in range(optimal_k)}
    for idx, label in enumerate(labels):
        clusters[label].append(grant_descriptions[idx])

    # 3. Generate Categories Dynamically (TF-IDF example)
    cluster_labels = {}
    for cluster_id, texts in clusters.items():
         categories = generate_categories_tfidf(texts) #Use TFIDF
         cluster_labels[cluster_id] = categories
         print(f"\nCluster {cluster_id} - Categories: {categories}")
         for text in texts:
             print(f" - {text}")
    return clusters, cluster_labels, labels

def find_optimal_clusters(embeddings, min_k=2, max_k=10):
    best_k = min_k
    best_score = -1
    for k in range(min_k, min(max_k + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"k={k}, silhouette score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def generate_categories_tfidf(cluster_texts, max_words=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)

    feature_names = vectorizer.get_feature_names_out()
    categories = []
    for i in range(len(cluster_texts)):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[-max_words:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        categories.extend(top_words)

    return list(set(categories))  # Remove duplicates

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8)

def extract_features_dynamic(description, categories):
    features = {}
    for category in categories:
        prompt = f"""Rate the following description from 0 to 1 on how relevant it is to the category: {category}.

        Description:
        {description}

        Rating (0-1):"""

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        try:
            rating = float(completion.choices[0].message.content.strip())
            features[category] = rating
        except (ValueError, TypeError):
            print(f"Warning: Could not convert rating for {category}. Setting to 0.0")
            features[category] = 0.0
        except Exception as e:
            print(f"Warning: Error getting rating for {category}: {e}. Setting to 0.0")
            features[category] = 0.0
    return features

def evaluate_project(project_proposal, clusters, cluster_labels, labels, grant_descriptions):
    project_embedding = get_openai_embedding(project_proposal)
    best_cluster_id = -1
    highest_similarity = -1
    best_cluster_categories = None # Store categories of the best cluster

    for cluster_id, texts in clusters.items():
        cluster_embeddings = [get_openai_embedding(text) for text in texts]
        cluster_embedding = np.mean(cluster_embeddings, axis=0)
        similarity = cosine_similarity(project_embedding, cluster_embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_cluster_id = cluster_id
            best_cluster_categories = cluster_labels[cluster_id] # Store the categories

    if best_cluster_id == -1:
        return None, None, "No suitable cluster found for the project proposal."

    # Use categories from the best cluster for BOTH project and grants
    relevant_categories = best_cluster_categories  # Categories for the project

    project_features = extract_features_dynamic(project_proposal, relevant_categories)

    # Convert to NumPy array, handling potential missing categories:
    project_vector = np.array([project_features.get(cat, 0) for cat in relevant_categories])


    grant_similarities = [] #List to store the similarities for each grant
    grant_features_list = [] #List to store all the grant features

    for grant_index in range(len(grant_descriptions)): #Loop through all the grants
        grant_categories = cluster_labels[labels[grant_index]] # Get the categories for the grant
        grant_features = extract_features_dynamic(grant_descriptions[grant_index], relevant_categories) #Use relevant_categories
        grant_vector = np.array([grant_features.get(cat, 0) for cat in relevant_categories]) # Same categories as project
        similarity_score = cosine_similarity(project_vector, grant_vector)
        grant_similarities.append(similarity_score) #Append to the list
        grant_features_list.append(grant_features) #Append to the list

    #Find the best matching grant
    best_match_index = np.argmax(grant_similarities)
    best_match_grant = grant_descriptions[best_match_index]
    best_match_features = grant_features_list[best_match_index]
    best_match_similarity = grant_similarities[best_match_index]

    return project_features, best_match_similarity, None, best_match_grant, best_match_features #Return best match grant and features