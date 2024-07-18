# SCRAPING:

#Scraping with requests
import requests
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
data = requests.get(standings_url) # Makes a request to the server and downloads the HTML of the server
#Parsing the HTML links with Beatiful Soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(data.text)
standings_table = soup.select('table.stats_table')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l] # Filters links ( Only squad links)
team_urls = [f"https://fbref.com{l}" for l in links] # Takes each of the lings and adds the https://fbref.com string to the beg of the link
#Getting Match Stats using pandas and requests
data = requests.get(team_urls[0])
import pandas as pd
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
#getting match shooting stats
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l] # Filters links, only gets the ones with shooting, have the element of "all_comps/shooting/"
data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]
shooting.head()
#Cleaning and merging scraped data with pandas
shooting.columns = shooting.columns.droplevel()
team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
team_data.head()
#Scraping data for mult seasons with a for loop
years = list(range(2022, 2020, -1)) # Starts with current seasons and then goes backwards
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
import time
for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(1)
len(all_matches)
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("matches.csv")

#AI PREDICTION:

import pandas as pd
matches = pd.read_csv("matches.csv", index_col=0)
matches.head()
# - INVESTIGATING "MISSING" DATA - The data was collected mid season, and some teams aren't fully recordered due to relegation or simply just not there
38*20*2 # Total games per season, 38 games per team, 20 teams, 2 seasons.
matches["team"].value_counts()
matches[matches["team"] == "Liverpool"].sort_values("date")
matches["round"].value_counts()
# - CLEANING THE DATA - No objects, changing categories, only floats.
matches.dtypes
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# - CREATING INITIAL MACHINE LEARNING MODEL - 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["data" <= '2022-01-01']]
test = matches[matches["date"] >= '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train ["target"]) # This is going to train a random forest model with predictors to predict the target, which is 0 for a loss/draw, and 3 for a win. 
preds = rf.predict(test[predictors]) # This generates predictions
# This part is to determine the accuracy of the model
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target], preds"])
acc 
#Checking what parts of the output is innacurate 
combined = pd.DataFram(dict(actual=test["target"], predction=preds))
pd.crosstab(index=combined["actual"], columns=combined["predicted"]) #This shows that most of the time with loss/draws we are more accurate than not, but when it comes to wins we were wrong about more than we were right
from sklearn.metrics import precision_score
precision_score(test["target"], preds)
# Now time to improve the precision with rolling averages
grouped_matches = matches.groupby("team")#This creates 1 data fram for every team in our data ( Liverpool, man city, burnly, man u, etc.)
#eg.
group = grouped_matches.get_group("Manchester City") # This gives out a single group
# This following code is for including team form as a predictor. Team form is a very important predictor because it is essentially just momentum. If you have a long stirng of losses, its hard to get a win compared to when you were already winning a good few games
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean() # The closed=left part just means that for eg. if you are in week 4, it will use the previous 3 weeks of data to help predict, discluding week 4.
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) # This excludes missing values, for example if you are in week 2, it won't use week 0 and week -1 because they dont exist
    return group
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
new_cols
rolling_averages(group, cols, new_cols) # Adds in extra info ( More columns ) from the previous 3 matches which would help the AI predict scores better, thus predicting L/D/W more accurately
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols)) # This applies the previous code to all teams now
matches_rolling = matches_rolling.droplevel('team') # Just makes the data table easier to read, this does not have to be included
matches_rolling.index = range(matches_rolling.shape[0]) #There were a lot of repeating values, as they would apply week 14 for example many many times which would create overlap, so now each match has its own unique number
#RETRAINING THE AI ( NEW PREDICTIONS )
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
precision
#preicision is now 62%
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
#COMBINING HOME ( STADIUM ) AND AWAY ( TRAVELING ) PREDICTIONS
# Some names aren't constant, as sometimes Wolves is called wolves, and other times its listed as it's official name of Wolverhampton Wanderers
class MissingDict(dict):
    __missing__ = lambda self, key:key
# Changing inconsistant names
map_values = {
    "Brighton and Hove Albion":"Brighton",
    "Mancehster United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}
mapping = MissingDict(**map_values)
mapping["West Ham United"]
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) #Makes sure that the predictions are consistant, so it doesn't predict on one side for arsenal 3-1 W, wheras it predicts a 1-5 demolishing of burnley ( However there are 2 games a season between 2 teams at the least)
merged
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()
# New accuracy: 67%

#Some Instructions:
#Use this in the notebook to view other matches: combined.head(__) Fiddle with the number inside the parentheses
#Use this: matches[matches["team"] == "____"], and change the team name in the underlined part to see team history and what data is available