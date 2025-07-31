# match_progression_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Data loading and preprocessing
@st.cache_data
def load_data():
    # Load datasets (update paths if needed)
    match = pd.read_csv("BPL_dataset_1.csv")
    delivery = pd.read_csv("BPL_deliveries_dataset_2.csv")
    
    # Preprocessing pipeline
    total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning']==1]
    match_df = match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')

    teams = [
        'Khulna Tigers', 'Rangpur Riders', 'Barishal Bulls',
        'Sylhet Strikers', 'Rajshahi Kings', 'Dhaka Dominators',
        'Comilla Victorians', 'Chattogram Challengers'
    ]
    
    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]
    match_df = match_df[['match_id','city','winner','total_runs']]
    
    delivery_df = match_df.merge(delivery, on='match_id')
    delivery_df = delivery_df[delivery_df['inning']==2]
    
    # Feature engineering
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])
    
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x=="0" else "1")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
    
    wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
    delivery_df['wickets'] = 10 - wickets
    
    delivery_df['crr'] = (delivery_df['current_score']*6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left']*6) / delivery_df['balls_left']
    
    def result(row):
        return 1 if row['batting_team'] == row['winner'] else 0
    
    delivery_df['result'] = delivery_df.apply(result, axis=1)
    
    final_df = delivery_df[[
        'match_id', 'batting_team', 'bowling_team', 'city', 
        'runs_left', 'balls_left', 'wickets', 'total_runs_x', 
        'crr', 'rrr', 'result'
    ]]
    
    final_df = final_df.sample(final_df.shape[0])
    final_df.dropna(inplace=True)
    final_df = final_df[final_df['balls_left'] != 0]
    
    return delivery_df, final_df

# Model training
@st.cache_resource
def train_model(final_df):
    x = final_df.iloc[:, 1:-1]  # Exclude match_id
    y = final_df.iloc[:, -1]    # result
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )
    
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), 
        ['batting_team', 'bowling_team', 'city']
    ], remainder='passthrough')
    
    pipe = Pipeline(steps=[
        ('step1', trf), 
        ('step2', LogisticRegression(solver='liblinear'))
    ])
    
    pipe.fit(x_train, y_train)
    return pipe

# Match progression visualization
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match.groupby('over').tail(1)  # Last ball of each over
    
    if match.empty:
        return pd.DataFrame(), 0
    
    temp_df = match[[
        'batting_team', 'bowling_team', 'city', 'runs_left', 
        'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'
    ]]
    
    temp_df = temp_df[temp_df['balls_left'] != 0]
    
    if temp_df.empty:
        return pd.DataFrame(), 0
    
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)
    
    target = temp_df['total_runs_x'].values[0]
    
    # Calculate runs scored in each over
    temp_df['runs_after_over'] = temp_df['runs_left'].shift(1, fill_value=target) - temp_df['runs_left']
    
    # Calculate wickets fallen in each over
    temp_df['wickets_in_over'] = temp_df['wickets'].shift(1, fill_value=10) - temp_df['wickets']
    
    return temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']], target

# Streamlit app
def main():
    st.title("🏏 Cricket Match Win Probability Predictor")
    st.markdown("""
    Visualize win probability progression throughout T20 matches
    """)
    
    # Load data and train model
    delivery_df, final_df = load_data()
    pipe = train_model(final_df)
    
    # Get available match IDs
    available_matches = delivery_df[['match_id', 'batting_team', 'bowling_team']].drop_duplicates()
    available_matches['label'] = available_matches.apply(
        lambda row: f"Match {row['match_id']}: {row['batting_team']} vs {row['bowling_team']}", 
        axis=1
    )
    
    # Match selection
    selected_match = st.selectbox(
        "Select a Match", 
        options=available_matches['label'].unique()
    )
    match_id = available_matches[available_matches['label'] == selected_match]['match_id'].values[0]
    
    # Generate visualization
    if st.button("Generate Match Progression"):
        with st.spinner("Generating visualization..."):
            temp_df, target = match_progression(delivery_df, match_id, pipe)
            
            if temp_df.empty:
                st.warning("No data available for this match")
                return
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Win/loss probability lines
            ax.plot(temp_df['end_of_over'], temp_df['win'], 
                    color='#4CAF50', linewidth=4, label='Win Probability')
            ax.plot(temp_df['end_of_over'], temp_df['lose'], 
                    color='#F44336', linewidth=4, label='Loss Probability')
            
            # Runs per over bars
            ax.bar(temp_df['end_of_over'], temp_df['runs_after_over'], 
                   alpha=0.7, color='#2196F3', label='Runs in Over')
            
            # Wickets markers
            ax.scatter(temp_df['end_of_over'], temp_df['wickets_in_over']*5, 
                       color='#FFC107', s=100, zorder=5, 
                       label='Wickets (scaled)')
            
            ax.set_title(f"Target: {target} Runs", fontsize=16)
            ax.set_xlabel("Over", fontsize=12)
            ax.set_ylabel("Probability/Runs", fontsize=12)
            ax.set_xticks(temp_df['end_of_over'])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            ax.set_ylim(0, 110)
            
            st.pyplot(fig)
            
            # Show data table
            st.subheader("Over-by-Over Statistics")
            st.dataframe(temp_df.style.format({
                'win': '{:.1f}%',
                'lose': '{:.1f}%',
                'runs_after_over': '{:.0f}',
                'wickets_in_over': '{:.0f}'
            }))

if __name__ == "__main__":
    main()
