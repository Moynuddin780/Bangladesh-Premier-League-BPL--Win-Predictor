# BPL Match Outcome Predictor

Welcome to the **BPL Match Outcome Predictor**! This project uses machine learning to predict the outcome of Bangladesh Premier League (BPL) cricket matches based on synthetic datasets. Built with Python, Pandas, Scikit-learn, and deployed via Streamlit, this repository features a Logistic Regression model pipeline and an interactive web app for real-time win probability predictions. Last updated on **Wednesday, July 30, 2025, at 07:12 AM +06**.

---

## Project Overview

This project predicts whether the chasing team will win or lose in BPL matches by analyzing data from the first and second innings. It calculates key metrics such as Current Run Rate (CRR), Required Run Rate (RRR), runs left, balls left, and remaining wickets, providing dynamic win/loss probabilities. The included Streamlit app allows users to input match details and visualize predictions instantly.

---

## Key Features

- **Data Processing**: Loads and cleans BPL match and delivery datasets.
- **Feature Engineering**: Computes CRR, RRR, runs left, balls left, and wickets dynamically.
- **Machine Learning**: Utilizes a Logistic Regression pipeline with OneHotEncoding for categorical variables.
- **Visualization**: Plots match progression with runs, wickets, and win/lose probabilities.
- **Web Application**: Interactive Streamlit app for real-time predictions.
- **Model Persistence**: Saves the trained model using Pickle.
- **Accuracy**: Achieves ~85.7% accuracy on the test set.

---

## Results

The model delivers robust predictions across various matches. Key visualizations include:
- **Match Progression Plot**: Tracks runs scored, wickets lost, and win probabilities per over.
- **Win Probability Trends**: Shows how probabilities evolve during a match.

Check out the sample visualization below:

![Match Progression Plot](path/to/match_progression_plot.png)  
*(Note: Replace `path/to/` with the actual file path of your generated plot.)*

---

## How to Run

### Locally

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/bpl-match-predictor.git
   ```

2. **Install Dependencies**  
   Ensure you have Python 3.x and install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**  
   Place the `BPL_dataset_1.csv` and `BPL_deliveries_dataset_2.csv` files in the `data/` directory or update the file paths in the script (e.g., `E:\Jupyter Programs\`).

4. **Run the Script**  
   Execute the main Python file to train the model and generate predictions:
   ```bash
   python main.py
   ```

5. **Launch the Web App**  
   Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

### Deployed (e.g., Heroku)

1. **Set Up Environment**  
   Create a `Procfile` with:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. **Configure Streamlit**  
   Create a `.streamlit/config.toml` file with:
   ```
   [server]
   port = $PORT
   enableCORS = false
   headless = true
   ```

3. **Deploy**  
   Follow your hosting platform's deployment instructions (e.g., Heroku) with the provided `requirements.txt`.

---

## Project Structure

```
bpl-match-predictor/
│
├── data/              # Contains BPL_dataset_1.csv and BPL_deliveries_dataset_2.csv
├── models/            # Saved model file (pipe.pkl)
├── images/            # Generated plots (e.g., match_progression_plot.png)
├── app.py             # Streamlit web application
├── main.py            # Main script to train the model
├── setup.sh           # Setup script for deployment
├── Procfile           # Deployment configuration
├── .streamlit/        # Streamlit configuration directory
│   └── config.toml    # Streamlit settings
├── requirements.txt   # List of dependencies
└── README.md          # This file!
```

---

##  Usage Example

### Python Script
To predict the outcome for a specific match (e.g., `match_id = 1`):
- Run the `match_progression` function with the trained pipeline.
- Visualize the results using the provided plotting code.

Sample output for `match_id = 1`:
- **Target**: 452 runs
- **Win Probability**: Starts at 14.6% and rises to 15.8% by the 19th over.

### Web App
- Select the batting team, bowling team, city, target score, current score, overs completed, and wickets lost.
- Click "Predict Win Probability" to see the win/loss percentages and a progress bar.

---

## Supported Teams

- Khulna Tigers
- Rangpur Riders
- Barishal Bulls
- Sylhet Strikers
- Rajshahi Kings
- Dhaka Dominators
- Comilla Victorians
- Chattogram Challengers

---

## Supported Cities

- Chattogram
- Khulna
- Barishal
- Rangpur
- Dhaka
- Sylhet

---

## Contributions

Feel free to fork this repository, submit issues, or create pull requests. Suggestions to improve model accuracy or enhance the web app are highly appreciated!

---

## Acknowledgments

- Thanks to the creators of the synthetic BPL dataset for the foundation.
- Inspired by the Python, Scikit-learn, and Streamlit communities for their invaluable resources.

---

Predict like a pro! 
*Created with Love by Moynuddin Al Masum 
