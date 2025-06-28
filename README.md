# Movie Recommendation System

A hybrid movie recommendation system built using content-based filtering and collaborative filtering techniques. This project combines machine learning algorithms to provide personalized movie recommendations based on user preferences and movie metadata.

## 🎯 Project Overview

This is **Project-4** of the Codec Technologies internship program - my first self mini-project after completing the 1st month training. The system implements a sophisticated recommendation engine that can suggest movies based on:
- **Movie Title**: Find similar movies to one you liked
- **Genre**: Discover movies in your favorite genres  
- **Hybrid Approach**: Combine both title and genre preferences for refined recommendations

## 📊 Dataset

The project uses the famous **TMDB 5000 Movie Dataset** from Kaggle, combined with MovieLens ratings data. This comprehensive dataset provides rich movie metadata and user interaction data.

### Dataset Structure

```
data/
├── credits.csv          # Cast and crew information (cast, crew, id)
├── keywords.csv         # Movie keywords (id, keywords)
├── links.csv           # Cross-reference IDs (movieId, imdbId, tmdbId)
├── links_small.csv     # Smaller version of links (movieId, imdbId, tmdbId)
├── movies_metadata.csv # Rich movie metadata (24 columns including genres, overview, revenue, etc.)
├── ratings.csv         # User ratings (userId, movieId, rating, timestamp)
└── ratings_small.csv   # Smaller version of ratings (userId, movieId, rating, timestamp)
```

### Key Data Fields

**movies_metadata.csv** contains:
- `adult`, `belongs_to_collection`, `budget`, `genres`
- `homepage`, `id`, `imdb_id`, `original_language`, `original_title`
- `overview`, `popularity`, `poster_path`, `production_companies`
- `production_countries`, `release_date`, `revenue`, `runtime`
- `spoken_languages`, `status`, `tagline`, `title`, `video`
- `vote_average`, `vote_count`

## 🛠️ Technology Stack

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
  - TfidfVectorizer for text feature extraction
  - Cosine similarity for content-based filtering
- **Surprise** - Collaborative filtering (SVD algorithm)
- **ast** - Abstract syntax tree for JSON parsing

## 🧠 Machine Learning Approach

### 1. Content-Based Filtering
- **Feature Engineering**: Creates a "soup" of text features combining:
  - Genres
  - Keywords  
  - Movie overview
  - Cast information
  - Crew information
- **Vectorization**: Uses TF-IDF to convert text features into numerical vectors
- **Similarity**: Employs cosine similarity to find movies with similar content

### 2. Collaborative Filtering
- **Algorithm**: Singular Value Decomposition (SVD)
- **Training**: Uses 100% of available ratings data
- **Validation**: 5-fold cross-validation for model evaluation
- **Metrics**: RMSE and MAE for performance measurement

### 3. Hybrid Recommendation
Combines both approaches to provide more accurate and diverse recommendations.

## 🚀 Features

### Core Functionality
1. **Similar Movie Recommendations**: Input a movie title to get 100 similar movies
2. **Genre-Based Recommendations**: Discover movies by genre preferences
3. **Hybrid Recommendations**: Combine movie title and genre for refined results
4. **Flexible Input**: Users can provide either movie name, genre, or both

### Model Capabilities
- Processes and cleans complex nested JSON data structures
- Handles missing data gracefully
- Scales to large datasets efficiently
- Provides fast similarity computations using optimized algorithms

## 📋 Usage

### Basic Recommendations

```python
# Get similar movies to a specific title
recommendations = get_recommendations("The Dark Knight")

# Get movies by genre
action_movies = recommend_by_genre("Action")

# Hybrid recommendation (both title and genre)
hybrid_recs = hybrid_recommendation("The Dark Knight", "Action")
```

### User Interaction
The system prompts users for:
1. **Movie Name** (optional): A movie they enjoyed
2. **Genre** (optional): Their preferred genre(s)

Based on the input, it returns up to 100 personalized movie recommendations.

## 📈 Model Performance

- **Cross-Validation**: 5-fold CV with RMSE and MAE metrics
- **Training Split**: 80% training, 20% testing (random split)
- **Content Similarity**: Linear kernel for efficient cosine similarity computation
- **Recommendation Count**: Top 100 movies per query

## 🔧 Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Project-4
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn scikit-surprise
```

3. **Run the notebook**
```bash
jupyter notebook recommendation_engine.ipynb
```

## 📁 Project Structure

```
Project-4/
├── README.md                           # Project documentation
├── data/                              # Dataset files
│   ├── credits.csv
│   ├── keywords.csv
│   ├── links.csv
│   ├── links_small.csv
│   ├── movies_metadata.csv
│   ├── ratings.csv
│   └── ratings_small.csv
└── notebooks/
    └── recommendation_engine.ipynb     # Main implementation
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Data Preprocessing**: Cleaning and preparing complex movie datasets
- **Feature Engineering**: Creating meaningful features from text and metadata
- **Machine Learning**: Implementing both content-based and collaborative filtering
- **Model Evaluation**: Using cross-validation and appropriate metrics
- **System Design**: Building a user-friendly recommendation interface

## 🏢 About Codec Technologies Internship

This project is part of the Codec Technologies internship program, serving as the first self-directed mini-project after completing the foundational training month. It showcases the practical application of machine learning concepts in building real-world recommendation systems.

## 🚀 Future Enhancements

- **Deep Learning Integration**: Implement neural collaborative filtering
- **Real-time Updates**: Add online learning capabilities
- **Advanced Features**: Include movie trailers, reviews, and social features
- **Web Interface**: Develop a user-friendly web application
- **Performance Optimization**: Implement approximate nearest neighbors for faster similarity search

## 📝 Model Output

The system provides recommendations in a structured format showing:
- Movie titles
- Associated genres
- Similarity scores (for content-based recommendations)

Example output for hybrid recommendation:
```
Top 10 Recommendations for "The Dark Knight" in "Action" genre:
1. Batman Begins - Action Crime Drama
2. The Dark Knight Rises - Action Crime Thriller  
3. Man of Steel - Action Adventure Fantasy
...
```

---

**Note**: This recommendation system is built for educational purposes as part of the Codec Technologies internship program and demonstrates fundamental concepts in machine learning and recommendation systems.