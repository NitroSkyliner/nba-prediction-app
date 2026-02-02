#Importing the necessary libraries
import pandas as pd
import nba_api as nba
import time as time  # Unnecessary, just use: import time
from nba_api.stats.static import teams;  # Remove semicolon (not needed in Python)


from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams;

def get_all_teams():
    all_teams = teams.get_teams()
    return all_teams


def fetch_games_for_one_team(team_id):
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    return games


def fetch_all_games(list_of_seasons):
    all_teams_list = get_all_teams()
    game_list = []
    for season in list_of_seasons:
        for team in all_teams_list:
            try:
                games_df = (leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team["id"],
                season_nullable=season
                )).get_data_frames()[0]
                game_list.append(games_df)
                time.sleep(1)
                print(f"✓ {team['full_name']}: {len(games_df)} games")
            except Exception as e:
                print(f"Error fetching {team['full_name']}: {e}")
                continue

    combined_df = pd.concat(game_list, ignore_index=True)
    #combined_df = combined_df.drop_duplicates(subset=['GAME_ID'])
    return combined_df
    
def save_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def main():
    years = ['2022-23', '2023-24']
    game_list = fetch_all_games(years)
    save_to_csv(game_list, '../data/nba_games.csv')
    print("Data saved to games.csv")




if __name__ == "__main__":
    main()
