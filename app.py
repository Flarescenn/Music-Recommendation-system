import streamlit as st
import pandas as pd

df = pd.read_csv('preprocessed_data.csv')
features = ['acousticness', 'danceability', 'energy', 'valence', 'tempo']

st.title("🎵 Music Recommendation System")

uploaded_file = st.file_uploader("📤 Upload a song CSV (1 row only)", type=["csv"])
if uploaded_file:
    user_song = pd.read_csv(uploaded_file)
    st.write("🎧 Uploaded Song Info:", user_song)
    input_song = user_song.iloc[0]
else:
    song_names = df['name'].unique()
    selected_song = st.selectbox("🎶 Or select a song:", song_names)
    input_song = df[df['name'] == selected_song].iloc[0]

def recommend_songs(input_song, df, features=features, top_n=5):
    df_filtered = df[df['name'] != input_song['name']].copy()
    df_filtered['similarity'] = df_filtered[features].sub(input_song[features].values).abs().sum(axis=1)
    recommendations = df_filtered.sort_values('similarity').head(top_n)
    return recommendations

if st.button("🎯 Get Recommendations"):
    try:
        recommendations = recommend_songs(input_song, df)
        st.subheader("🎵 Top 5 Recommended Songs:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"### {row['name']} - *{row['artist']}*")
            for feature in features:
                st.write(f"**{feature.capitalize()}**: {row[feature]}")
            st.write("---")
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
