import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

# Custom CSS for advanced styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Roboto+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto Mono', monospace;
}

h1 {
    font-family: 'Press Start 2P', cursive;
    color: #00ff88 !important;
    text-shadow: 0 0 10px #00ff8888;
}

.stTextInput>div>div>input {
    background-color: #1a1a2e;
    color: #00ff88;
    border: 2px solid #00ff88;
    border-radius: 5px;
    padding: 10px;
}

.stButton>button {
    background: linear-gradient(45deg, #00ff88, #00ccff);
    color: #000 !important;
    border: none;
    border-radius: 25px;
    padding: 12px 24px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00ff88;
}

.custom-card {
    background: #16213e !important;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border-left: 5px solid #00ff88;
    transition: 0.3s;
}

.custom-card:hover {
    transform: translateX(10px);
    box-shadow: 0 0 20px #00ff8833;
}
</style>
""", unsafe_allow_html=True)

# Load and prepare data
df = pd.read_csv('steam_updated.csv')
df = df[['name', 'genres']].dropna().drop_duplicates(subset='name')
df = df.reset_index(drop=True)
df['name_lower'] = df['name'].str.lower()

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['name_lower']).drop_duplicates()

def get_closest_match(game_name, game_list):
    match, score = process.extractOne(game_name.lower(), game_list)
    return match if score >= 60 else None

def recommend_games(game_name, top_n=5):
    idx = indices.get(game_name.lower())
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[game_indices].tolist()

# Modern UI Layout
st.title("🔮 CYBER RECOMMENDER 3000")

# Animated header
with st.container():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h3 style="color: #00ccff; font-family: 'Press Start 2P', cursive;">
            NEXT-GEN GAME DISCOVERY ENGINE
        </h3>
    </div>
    """, unsafe_allow_html=True)

# Search section with glitch effect
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("", placeholder="ENTER GAME NAME...")
    with col2:
        st.markdown("<div style='height: 52px'></div>", unsafe_allow_html=True)
        if st.button("🚀 LAUNCH RECOMMENDATIONS"):
            st.balloons()

if user_input:
    game_list = df['name_lower'].tolist()
    closest_match = get_closest_match(user_input, game_list)

    if closest_match:
        original_name = df[df['name_lower'] == closest_match]['name'].values[0]
        recommendations = recommend_games(original_name)
        genre = df[df['name_lower'] == closest_match]['genres'].values[0]
        genre_display = genre if genre.strip() else "CLASSIFIED"

        # Result display
        with st.container():
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #00ff88; margin-bottom: 10px;">🔍 SEARCH RESULTS</h4>
                <p style="font-size: 1.2em; margin: 0;">
                    🎮 <strong>{original_name}</strong><br>
                    🏷️ GENRE: {genre_display}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if recommendations:
            st.markdown("""
            <div style="margin: 30px 0 10px 0;">
                <h3 style="color: #00ccff; border-bottom: 2px solid #00ff88; padding-bottom: 5px;">
                    RECOMMENDED TITLES
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, game in enumerate(recommendations, start=1):
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; align-items: center;">
                        <div style="font-size: 1.5em; margin-right: 15px; color: #00ff88;">#{i}</div>
                        <div style="font-size: 1.2em;">{game}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ NO SIGNAL FOUND - TRY ANOTHER TITLE")

    else:
        st.error("🛑 INPUT ERROR - VERIFY GAME TITLE")

# Add neon border effect
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(45deg, #0f0c29, #1a1a2e, #16213e);
    }
</style>
""", unsafe_allow_html=True)
