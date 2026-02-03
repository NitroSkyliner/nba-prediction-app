# NBA Game Prediction Tool

#### Video Demo: https://www.youtube.com/watch?v=1FrbCZRw7_o

#### Description:

This project is a full-stack web application that predicts the outcome of NBA basketball games using machine learning . The application analyzes team statistics like field goal percentage, rebounds, assists, steals, blocks, and turnovers to determine which team is more likely to win a matchup. Built with a Flask backend serving a trained machine learning model and a React frontend providing an intuitive user interface, this tool demonstrates the practical application of data science in sports analytics.

This project was built out of a passion for both basketball analytics and data science. As an NBA fan (GSW fan), I've been curious about what factors truly determine game outcomes beyond just team names and star players. This project solves the problem of making informed predictions about game outcomes by analyzing concrete statistical data rather than relying on gut feelings or biases. What makes it unique is its combination of high accuracy (88.27%), a complete full-stack architecture, and a user-friendly interface that makes machine learning predictions accessible to anyone, whether they're a casual fan or a serious sports analyst.

The app not only predicts winners but also provides confidence levels and win probabilities for both teams, helping users understand not just *who* will win, but *how likely* that outcome is based on the statistical matchup.

---

## Project Overview

The NBA Game Prediction Tool works by taking historical game data from the 2022-2024 NBA seasons, processing it into a format suitable for machine learning, training multiple models to find the best predictor, and then serving those predictions through a web API that powers an interactive React frontend. Users can input statistics for a hypothetical matchup between any two teams, and the model will instantly predict the winner with a confidence percentage.

The technology stack includes Python with Flask for the backend API, scikit-learn for machine learning, React for the frontend interface, and axios for API communication. My approach followed a clear data science pipeline: first, I collected 2,766 historical NBA games using the nba_api library; second, I processed this raw data into meaningful features that capture team performance; third, I trained and compared three different machine learning models (Logistic Regression, Random Forest, and Gradient Boosting); fourth, I built a REST API to serve predictions from the best model; and finally, I created an intuitive React interface where users can interact with the predictions in real-time.

---

## Technical Architecture

### Backend (Python/Flask)

I chose Flask for the backend because it's lightweight, perfect for machine learning projects, and integrates seamlessly with Python's data science ecosystem. Unlike heavier frameworks like Django, Flask gives me the flexibility to structure the application exactly how I need it without unnecessary overhead. The API is simple yet powerful, with three main endpoints that serve different purposes.

The `/health` endpoint provides a quick check that the server is running and responsive. The `/model-info` endpoint returns metadata about the loaded machine learning model, including its type, accuracy, and the number of features it uses. The core `/predict` endpoint accepts team statistics in JSON format, processes them through the same feature engineering pipeline used during training, runs them through the trained Gradient Boosting model, and returns predictions with confidence levels.

The Flask app connects to the ML model through joblib serialization. When the server starts, it loads the pre-trained model from a .pkl file into memory, making predictions extremely fast since there's no need to retrain for each request. The application validates all incoming data to ensure it matches the expected format and gracefully handles errors with appropriate HTTP status codes and error messages.

### Machine Learning Model

The foundation of this project is a carefully trained machine learning model built on 2,766 NBA games from the 2022 and 2023 regular seasons. I chose this time period to balance having enough data for robust training while keeping the model focused on recent NBA dynamics, as team strategies and playing styles evolve over time. Using older data might capture outdated patterns that don't reflect how modern basketball is played.

Data processing was a critical step that transformed raw game logs into meaningful training data. The original data came as single rows per team per game, but I needed to convert this into matchup format where each row represents a complete game with both teams' statistics. I calculated difference features like FG% differential (home team's field goal percentage minus away team's), rebound differential, and assist differential because these capture not just how well each team performed, but how they performed *relative to each other*. This relative comparison proved crucial for prediction accuracy.

I experimented with three different machine learning algorithms to find the best predictor. Logistic Regression served as a simple baseline and achieved around 72% accuracy. Random Forest improved this to about 85% by capturing non-linear relationships between features. Gradient Boosting emerged as the clear winner with 88.27% accuracy. This model works by building trees sequentially, where each new tree corrects the errors of previous trees, creating a powerful ensemble predictor. The 88.27% accuracy is actually excellent for NBA game prediction because basketball has inherent unpredictability – upsets happen, injuries occur, and momentum swings games. Most simple prediction models struggle to break 70%.

Feature importance analysis revealed fascinating insights about what actually drives game outcomes. Field goal percentage differential emerged as the single most important factor, accounting for approximately 50% of the model's predictive power. This makes intuitive sense: the team that shoots more efficiently typically scores more points. Rebound differential came in second, reflecting basketball wisdom that "controlling the boards controls the game." Turnover differential also proved significant, as protecting the ball prevents giving opponents easy scoring opportunities. Interestingly, while assists, steals, and blocks contribute to predictions, their individual impact is smaller than the "big three" of shooting, rebounding, and ball security.

I deliberately chose not to include player-specific data (like whether specific stars are playing) to keep the scope manageable and focus on team-level statistics. This also makes the tool more practical since team stats are easier to project or input than tracking individual player availability.

### Frontend (React)

React was the natural choice for the frontend because it's the modern standard for building interactive user interfaces with component-based architecture. Its virtual DOM ensures smooth, responsive updates when users enter statistics or receive predictions. React's state management makes it simple to handle user inputs, loading states, and API responses without complex DOM manipulation.

The UI follows a clean, intuitive flow: users see two side-by-side sections for entering home team and away team statistics. Each section contains eight input fields for the key metrics (field goal percentage, 3-point percentage, free throw percentage, rebounds, assists, steals, blocks, and turnovers). After entering stats, users click the prominent "PREDICT WINNER" button, which triggers an API call to the Flask backend. The application provides immediate visual feedback with a loading state, then displays results in an attractive card showing the predicted winner, overall confidence level, and individual win probabilities for both teams.

Design choices prioritized clarity and usability. I used gradient backgrounds and smooth animations to create a modern, engaging feel without overwhelming users. Progress bars visually represent win probabilities, making it instantly clear how confident the model is in its prediction. Color coding distinguishes home (purple gradient) from away (pink gradient) teams. Default values populate all fields so users can test the application immediately without hunting for realistic statistics. The responsive design ensures the tool works beautifully on both desktop and mobile devices.

---

## File Structure

### Backend Files

**`backend/data_fetch.py`**

This file handles the initial data collection from the NBA API. It uses the nba_api library to fetch complete game logs for the 2022 and 2023 NBA regular seasons, downloading information about every game played during those years. The script processes the API responses, organizing them into a clean pandas DataFrame with all the statistics we need (field goal percentage, rebounds, assists, etc.). Finally, it saves this raw data to a CSV file for processing. I needed this file because manually collecting thousands of game statistics would be impractical, and the NBA API provides reliable, official data directly from the league.

**`backend/data_processing.py`**

This file transforms the raw game data into a format suitable for machine learning. It loads the CSV created by data_fetch.py, then performs the critical conversion from single-team rows to matchup rows. For each game, it combines the home team's statistics and away team's statistics into a single row representing the complete matchup. It then calculates the all-important difference features (like field goal percentage differential, rebound differential, and assist differential) that help the model understand relative team performance. The processed data is saved to a new CSV file that becomes the training dataset. This preprocessing step was essential because machine learning models need to learn from complete game contexts, not isolated team performances.

**`backend/model.py`**

This is where the machine learning magic happens. The file loads the processed game data and splits it into training and testing sets (80/20 split). It then trains three different classification algorithms: Logistic Regression as a baseline, Random Forest for ensemble learning, and Gradient Boosting for advanced sequential tree building. Each model is evaluated on the held-out test set to measure accuracy. The script compares all three models and identifies Gradient Boosting as the winner with 88.27% accuracy. It then saves this trained model to a .pkl file using joblib, making it ready for deployment in the Flask API. Additionally, it displays feature importance rankings so we can understand which statistics matter most for predictions. This file represents the core data science work of the entire project.

**`backend/app.py`**

This file implements the Flask web server that serves predictions to the frontend. When the server starts, it loads the trained Gradient Boosting model from the .pkl file into memory. It then defines three REST API endpoints: `/health` for basic server status checks, `/model-info` for model metadata, and `/predict` for actual game predictions. The `/predict` endpoint receives team statistics via POST request, validates the input data, performs the same feature engineering used during training (calculating differentials), feeds the features through the model, and returns predictions as JSON including the predicted winner, confidence level, and individual win probabilities. CORS is enabled so the React frontend can make cross-origin requests without browser security blocking them.

**`backend/requirements.txt`**

Lists all Python dependencies needed to run the backend: Flask for the web server, flask-cors for cross-origin requests, scikit-learn for machine learning, pandas and numpy for data processing, joblib for model serialization, and nba_api for data collection.

### Frontend Files

**`frontend/src/App.js`**

This is the main React component that powers the entire user interface. It manages state for home team statistics, away team statistics, prediction results, loading status, and error messages using React hooks. The component defines handler functions for updating statistics as users type, and a prediction function that constructs the API request, sends it to the Flask backend using axios, and processes the response. The render method creates the two-column layout with input forms for both teams, the predict button, and conditional rendering for results or errors. It handles all user interactions and API communication, serving as the brain of the frontend application.

**`frontend/src/App.css`**

Contains all styling for the application including the purple-pink gradient background, team section layouts, input field styling with focus effects, the prominent predict button with hover animations, result card design with probability bars, and responsive media queries for mobile devices. The CSS creates the modern, polished look that makes the application feel professional and engaging.

### Data Files

**`data/nba_games.csv`**

Raw NBA game data fetched from the API containing 2,766 games from the 2022 and 2023 regular seasons, with each row representing one team's performance in one game.

**`data/processed_games.csv`**

Processed matchup data ready for machine learning training, with 2,766 rows where each row represents a complete game with both teams' statistics and calculated differential features.

**`data/nba_model.pkl`**

The trained Gradient Boosting model serialized and saved in Python's pickle format, ready to be loaded by the Flask API for making predictions.

---

## Design Decisions

**Decision 1: Why Gradient Boosting?**

After training and comparing three different models, Gradient Boosting emerged as the clear winner with 88.27% accuracy compared to Logistic Regression's ~72% and Random Forest's ~85%. While Gradient Boosting is more computationally complex and takes longer to train, the significant accuracy improvement made it worth the tradeoff. For a prediction tool, accuracy is paramount – users want reliable predictions they can trust. The boosting algorithm's ability to sequentially correct errors and capture complex non-linear relationships between features (like how field goal percentage interacts with rebounding) gives it an edge in modeling basketball's intricate dynamics.

**Decision 2: Why These Features?**

I chose field goal percentage, three-point percentage, free throw percentage, rebounds, assists, steals, blocks, and turnovers because these statistics comprehensively capture team performance across all major aspects of basketball: scoring efficiency, ball control, defense, and rebounding. The feature importance analysis validated this choice, showing field goal percentage differential as 50% of predictive power. I deliberately calculated difference features (home stat minus away stat) because the model needs to understand *relative* performance, not absolute values. A team averaging 45 rebounds might dominate against a team averaging 40, but struggle against one averaging 50. The differentials capture this crucial context.

I specifically chose not to include player-specific data (like whether star players are injured or resting) for several reasons. First, it would dramatically increase complexity, requiring tracking of rosters, injury reports, and individual player statistics. Second, it would make the tool less practical for users who just want to compare team-level stats. Third, team statistics already implicitly capture player contributions – if a star is out, team field goal percentage typically drops. This decision kept the scope manageable while still achieving excellent accuracy.

**Decision 3: Why Two Seasons of Data?**

Using exactly two seasons (2022 and 2023) strikes the optimal balance between having enough data for robust model training and keeping predictions relevant to current NBA dynamics. With 2,766 games, the model has sufficient examples to learn meaningful patterns without overfitting. Using just one season (around 1,383 games) risked insufficient data diversity, potentially causing the model to memorize specific matchups rather than learning generalizable patterns. However, going back further than two seasons would introduce data from a different era of basketball – strategies evolve, rules change slightly, and team compositions shift dramatically. The two-season window ensures the model learns from recent, relevant basketball while having enough data to be statistically robust.

**Decision 4: UI/UX Choices**

The progress bars for win probabilities provide instant visual understanding of prediction confidence. Numbers alone (like "73.45% vs 26.55%") require users to process and compare values mentally, but colored bars of different lengths create an immediate visual impression of how lopsided or close the prediction is. Default values populate all input fields so users can click "PREDICT WINNER" immediately upon loading the page, reducing friction and encouraging experimentation. Color coding (purple gradient for home, pink gradient for away) helps users mentally separate the two teams and quickly identify results. The large, prominent predict button with hover effects and loading states makes the primary action obvious and provides clear feedback about system status.

**Decision 5: API Design**

I created separate `/predict` and `/predict-simple` endpoints to serve different use cases. The `/predict` endpoint requires full team statistics and returns detailed predictions with confidence levels and probabilities – this is for serious use with real data. The `/predict-simple` endpoint accepts just team names and uses dummy average statistics, making it perfect for quick testing and development without needing to gather actual stats. This separation keeps the main endpoint clean and focused while providing a convenient development tool. The `/health` and `/model-info` endpoints follow REST API best practices, enabling monitoring and debugging of the deployed application.

---

## Challenges Faced

The biggest technical challenge was ensuring perfect alignment between features used during training and features sent for prediction. During early testing, predictions were occasionally nonsensical because I was calculating differentials in slightly different orders or using different rounding. I solved this by creating a strict feature engineering function used by both the training script and the API endpoint, ensuring absolute consistency. Another significant challenge was dealing with data quality issues like duplicate games and missing statistics in the raw NBA API data. I had to implement robust cleaning procedures to filter out invalid entries without losing valuable training examples.

Setting up CORS properly for React-Flask communication took some troubleshooting. Initially, the React app couldn't make requests to the Flask backend because browsers block cross-origin requests by default. After adding flask-cors and understanding CORS headers, the integration worked smoothly. Achieving 88.27% accuracy required multiple iterations of feature engineering and hyperparameter tuning – my first models were only around 78% accurate. Through systematic experimentation with different feature combinations and model architectures, I gradually improved performance to the final 88% threshold. Learning to properly serialize and deserialize the trained model with joblib was also a new skill I had to develop to make the Flask API work correctly.

---

## How to Run

### Prerequisites
```
- Python 3.8+
- Node.js 14+
- npm or yarn
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Server will start on http://localhost:5000

### Frontend Setup
```bash
cd frontend
npm install
npm start
```
App will open at http://localhost:3000

### Access
- Frontend Interface: http://localhost:3000
- Backend API: http://localhost:5000
- Health Check: http://localhost:5000/health
- Model Info: http://localhost:5000/model-info

---

## Future Improvements

If I had more time, I would implement several enhancements to make this tool even more powerful. First, I'd integrate real-time data fetching to automatically pull current season statistics, eliminating manual entry and ensuring predictions reflect the latest team performance. Second, I'd add dropdown menus for team selection where users just pick two teams and the application automatically populates their recent average statistics. Third, incorporating player injury data would significantly improve accuracy since missing star players dramatically affects outcomes. Fourth, I'd implement historical prediction tracking to let users see how the model has performed on recent games, building trust through transparency. Finally, deploying the application to a cloud platform like Heroku or AWS would make it publicly accessible, allowing NBA fans worldwide to use the tool without needing to run it locally.

Additional technical improvements could include building a more sophisticated model that considers home court advantage more explicitly, incorporating betting lines as a feature to capture market wisdom, or using deep learning approaches like neural networks that might capture even more complex patterns in the data.

---

## Conclusion

Building this NBA Game Prediction Tool taught me invaluable lessons about the entire data science pipeline, from data collection through model deployment. I learned how to work with real-world APIs, clean messy data, engineer meaningful features, compare multiple machine learning algorithms, and integrate predictions into a full-stack web application. I'm particularly proud of achieving 88.27% accuracy, which required careful feature engineering and model selection, and creating a beautiful, intuitive interface that makes machine learning accessible to non-technical users.

This project demonstrates core CS50 concepts including algorithmic thinking (choosing the right ML algorithm), data structures (organizing game statistics efficiently), web development (Flask and React integration), and problem-solving (overcoming challenges in data processing and API design). More broadly, it shows how computer science can be applied to real-world problems – in this case, helping basketball fans make informed predictions about their favorite sport. The combination of data science, backend development, and frontend design showcases the multidisciplinary nature of modern software engineering.

---

## Acknowledgments

- CS50 course staff for providing the foundation in computer science that made this project possible
- nba_api library creators for providing clean, reliable access to NBA statistics
- scikit-learn documentation and community for comprehensive machine learning resources
- The NBA for maintaining detailed statistical records that make projects like this feasible
- Basketball analytics pioneers who demonstrated that sports outcomes can be predicted with data

---

**Built with ❤️ for CS50 Final Project**