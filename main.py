import pandas as pd #Basic Data frame to work with data in a table format
from sklearn.feature_extraction.text import TfidfVectorizer # Converts Text data to numbers which can be understood by the machine
from sklearn.metrics.pairwise import cosine_similarity  # Measures the similarity between two vectors 
from thefuzz import process  # Fuzzy string matching to find similar game names


df = pd.read_csv(r'C:\Projects\Game Recommender\data\steam_updated.csv')

df = df[['name', 'genres']]
df.dropna(inplace=True) # removes rows with missing generes
df.drop_duplicates(subset='name', inplace=True) # removes duplicate games based on name

tfidf = TfidfVectorizer(stop_words='english') # term frequency-inverse document frequency vectorizer, stop_words removes common words that do not contribute to the meaning of the text
tfidf_matrix = tfidf.fit_transform(df['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) #1 --> identical, 0 --> no similarity

# creates a series with game names as index and their corresponding indices in the dataframe
indices = pd.Series(df.index, index=df['name']).drop_duplicates() # you enter the name of the game and it will return the row number 

#recommender function
def recommend_games(game_name, top_n=5):
    idx = indices.get(game_name)
    if idx is None:
        return f"Game '{game_name}' not found in the dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]

    return df['name'].iloc[game_indices].tolist()

def get_closest_match(game_name, game_list):
    match, score = process.extractOne(game_name, game_list)
    if score >= 60:  # Minimum confidence threshold
        return match
    else:
        return None

#Test the recommender function
if __name__ == "__main__":
    game = input("Enter a game name: ")
    
    # Fuzzy match the game name
    game_list = df['name'].tolist()
    closest_game = get_closest_match(game, game_list)

    if closest_game:
        print(f"\nDid you mean: {closest_game}?")
        print("\nRecommended games:")
        recommendations = recommend_games(closest_game)
        for i, title in enumerate(recommendations, start=1):
            print(f"{i}. {title}")
    else:
        print(f"Could not find a close match for '{game}'. Please check the spelling.")

