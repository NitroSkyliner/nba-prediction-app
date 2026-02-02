import pandas as pd
import numpy as np

def parse_matchup(matchup_string):
    if "vs." in matchup_string:
        teams = matchup_string.split(" vs. ")
        home_team = teams[0].strip()
        away_team = teams[1].strip()
    else:
        if "@" in matchup_string:
            teams = matchup_string.split("@")
            away_team = teams[0].strip()  # Team before @ is AWAY
            home_team = teams[1].strip()  # Team after @ is HOME

    return home_team, away_team

def create_matchup_row(game_group):
    row1 = game_group.iloc[0]
    row2 = game_group.iloc[1]

    home1, away1 = parse_matchup(row1['MATCHUP'])
    home2, away2 = parse_matchup(row2['MATCHUP'])

    
    # Determine which row is home team
    if row1['TEAM_ABBREVIATION'] == home1:
        home_row = row1
        away_row = row2
    else:
        home_row = row2
        away_row = row1
    
    # Create new combined row
    matchup_data = {
        'game_id': home_row['GAME_ID'],
        'game_date': home_row['GAME_DATE'],
        'season': home_row['SEASON_ID'],
        
        # Home team info
        'home_team': home_row['TEAM_NAME'],
        'home_team_abbr': home_row['TEAM_ABBREVIATION'],
        'home_pts': home_row['PTS'],
        'home_fg_pct': home_row['FG_PCT'],
        'home_fg3_pct': home_row['FG3_PCT'],
        'home_ft_pct': home_row['FT_PCT'],
        'home_reb': home_row['REB'],
        'home_ast': home_row['AST'],
        'home_stl': home_row['STL'],
        'home_blk': home_row['BLK'],
        'home_tov': home_row['TOV'],
        
        # Away team info
        'away_team': away_row['TEAM_NAME'],
        'away_team_abbr': away_row['TEAM_ABBREVIATION'],
        'away_pts': away_row['PTS'],
        'away_fg_pct': away_row['FG_PCT'],
        'away_fg3_pct': away_row['FG3_PCT'],
        'away_ft_pct': away_row['FT_PCT'],
        'away_reb': away_row['REB'],
        'away_ast': away_row['AST'],
        'away_stl': away_row['STL'],
        'away_blk': away_row['BLK'],
        'away_tov': away_row['TOV'],
        
        # Target variable (what we're predicting)
        'home_win': 1 if home_row['WL'] == 'W' else 0
    }
    
    return matchup_data


def process_games(input_csv_path):

    
    # Step 1: Load raw data
    print("Loading data...")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Step 2: Group by GAME_ID
    print("Processing matchups...")
    game_groups = df.groupby('GAME_ID')
    
    processed_games = []  # Empty list to store processed rows
    
    # Step 3: Process each game
    for game_id, game_group in game_groups:
        if len(game_group) != 2:
            print(f"Skipping game {game_id} - has {len(game_group)} row(s)")
            continue
        
        try:
            processed_row = create_matchup_row(game_group)
            processed_games.append(processed_row)
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            continue
    # Step 4: Create DataFrame
    processed_df = pd.DataFrame(processed_games)
    print(f"Created {len(processed_df)} rows")
    
    # Step 4: Create DataFrame
    processed_df = pd.DataFrame(processed_games)
    print(f"Created {len(processed_df)} rows")

    # Step 5: Add calculated features  ← ADD THIS
    print("Adding features...")
    processed_df['point_diff'] = processed_df['home_pts'] - processed_df['away_pts']
    processed_df['fg_pct_diff'] = processed_df['home_fg_pct'] - processed_df['away_fg_pct']
    processed_df['reb_diff'] = processed_df['home_reb'] - processed_df['away_reb']
    processed_df['ast_diff'] = processed_df['home_ast'] - processed_df['away_ast']

    # Step 6: Remove rows with missing data  ← ADD THIS
    processed_df = processed_df.dropna()
    print(f"After removing missing data: {len(processed_df)} rows")

    return processed_df

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}") 

def main(): 
    # Define file paths
    input_file = '../data/nba_games.csv'
    output_file = '../data/processed_games.csv'
    
    # Process the data
    processed_df = process_games(input_file)
    
    # Show sample of processed data
    print(processed_df.head())
    
    
    print("\nColumns in processed data:")

    print(list(processed_df.columns))

    print(f"\nHome team win rate: {processed_df['home_win'].mean() * 100:.1f}%")

    # Save processed data
    save_processed_data(processed_df, output_file)

if __name__ == "__main__":
    main()