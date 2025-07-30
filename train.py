import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load match and delivery datasets
match = pd.read_csv(r"E:\Jupyter Programs\BPL_dataset_1.csv")
delivery = pd.read_csv(r"E:\Jupyter Programs\BPL_deliveries_dataset_2.csv")

# Calculate total runs for each match in the 1st inning
total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# Filter only valid teams
teams = [
    'Khulna Tigers', 
    'Rangpur Riders', 
    'Barishal Bulls',
    'Sylhet Strikers', 
    'Rajshahi Kings', 
    'Dhaka Dominators',
    'Comilla Victorians', 
    'Chattogram Challengers'
]
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# Keep necessary columns and merge with delivery data
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
delivery_df = match_df.merge(delivery, on='match_id')

# Keep only 2nd innings data
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Calculate current cumulative score, runs left, and balls left
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

# Calculate remaining wickets
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets

# Compute CRR and RRR
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

# Create binary result column
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0
delivery_df['result'] = delivery_df.apply(result, axis=1)

# Final feature set for modeling
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 
                        'total_runs_x', 'crr', 'rrr', 'result']]
final_df = final_df.sample(final_df.shape[0])
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

# Split data into features and target
x = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Create and train the pipeline with Logistic Regression
trf = ColumnTransformer([('trf', OneHotEncoder(sparse_output=False, drop='first'), 
                         ['batting_team', 'bowling_team', 'city'])], remainder='passthrough')
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])
pipe.fit(x_train, y_train)

# Evaluate accuracy
y_pred = pipe.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained pipeline
pickle.dump(pipe, open('pipe.pkl', 'wb'))
