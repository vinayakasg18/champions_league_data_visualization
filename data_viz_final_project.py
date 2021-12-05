# %% [markdown]
# 

# %% [markdown]
# # Data Visualization Project

# %% [markdown]
# ## Visualizing the best soccer team in the world.

# %% [markdown]
# ### Team: Visualizing the champions
# - Manisha Chandran (manchand)
# - Suraj Punjabi (spunjab)
# - Vinayaka Gadag (vgadag)

# %%
# load csv data
import pandas as pd
import numpy as np
og_data = pd.read_excel('champions-league-data.xlsx', engine='openpyxl')

# %%
# original dataframe
og_df = pd.DataFrame(og_data)

# %%
og_df.head()

# %% [markdown]
# #### We want to see total number of wins by a club and dropping the unwanted columns to make it easier for analysis.

# %%
# Drop unwanted columns for analysis
req_df = og_data.drop(columns=['scorer 1', 'scorer 2', 'scorer 3', 'scorer 4', 'scorer 5', 'scorer 6', 'scorer 7', 'opponent scorer 1', 'opponent scorer 2', 'opponent scorer 3'])
req_df.head()

# %% [markdown]
# #### Getting the total win count by value_counts() this gives a series dataframe with the relevant information for further analysis and visualization. By getting the total wins we can ....

# %%
# get the total number of wins by club
s = req_df['Winner'].value_counts()

# %%
winx = {}
for i, j in s.items():
  winx[i] = j

# %%
windf = pd.DataFrame(winx.items(), columns=['Team', 'Wins']) 

# %%
windf.head()

# %% [markdown]
# #### Write something here for finalists

# %%
# Number of lossess by club
f = req_df['Finalist'].value_counts()

# %%
finx = {}
for i, j in f.items():
  finx[i] = j

# %%
findf = pd.DataFrame(finx.items(), columns=['Finalist_Team', 'Losses']) 

# %%
findf.head()

# %% [markdown]
# #### Tell something about the total number of loses

# %%
# We want to visualize the number of wins by a club
import plotly.express as px
fig = px.bar(windf, x="Team", y="Wins", title="No of wins by team")
fig.show()

# %% [markdown]
# #### The above plot shows the number of wins against the teams. We can see that the club 'Real Madrid' have most number wins 13 and Hamburg and few other teams have just the one win. By visualizing the plot we can intepret that the 'Real Madrid' have dominated the league over the years. This can be visually verified by looking at the second most wins by a team 'Milan' being only 7 which is just above half wins of 'Real madrid'.

# %% [markdown]
# #### Talk something about this visualization. (Wins)

# %%
# We want to visualize the win % by a club in entire competition
import plotly.express as px

fig = px.pie(windf, values=windf['Wins'], names=windf['Team'], title='Win percentage per club')
fig.show()

# %% [markdown]
# #### To support the previous visualization method. If you look the win percentage shared between the clubs 'Real Madrid' again stands out with hisghest win percent of almost 20% and 'Milan' beign the second most with almost 11%.

# %%
# How the wins are scattered around the number of matches
import plotly.express as px
df = windf
fig = px.scatter(x=windf['Team'], y=windf['Wins'], color=windf['Team'])
fig.show()

# %% [markdown]
# #### This is another way of representing the number of wins using different visualization method(Scatter).

# %%
wins_by_country = req_df.groupby('Winner Country').size().sort_values(ascending=False).reset_index(name='Total Wins')
wins_by_country.head()

# %% [markdown]
# #### We have seen which clubs have won most titles. Now, let's see the number of wins by country.

# %%
import plotly.express as px
fig = px.line(wins_by_country, x="Winner Country", y="Total Wins", title='Total wins by country')
fig.show()
# when we hover over the graph we can see the country that won and total number of times it won

# %% [markdown]
# #### By looking at the visulaizaiton we can see that the most number of wins by spain are higher comparing to other countries and 'Real Madrid' and 'Barcelona'have contibuted the most number of wins.

# %%
f = req_df['Finalist'].value_counts()

# %%
finx = {}
for i, j in f.items():
  finx[i] = j

# %%
findf = pd.DataFrame(finx.items(), columns=['Finalist_Team', 'Losses']) 

# %%
import plotly.express as px
fig = px.scatter(x=findf['Finalist_Team'], y=findf['Losses'], color=findf['Finalist_Team'])
fig.show()

# %% [markdown]
# #### The above visualization method shows the number of times the club has lost in the finals. Juventus have lost total of 7 times.

# %% [markdown]
# #### Exploring the data, total number of final appearences by each club.

# %%
finalist_count = winx.copy()
for key, value in finx.items():
  if key in finalist_count.keys():
    finalist_count[key] = value + finalist_count[key]
  else:
    finalist_count[key] = value

# %%
finalist_count_df = pd.DataFrame(finalist_count.items(), columns=['Team', 'finals'])

# %%
finalist_count_df.head()

# %%
import plotly.express as px

fig = px.histogram(finalist_count_df, x="finals", color="Team")
fig.show()

# %% [markdown]
# #### We have seen the number of wins by a club. Now, let's see how many times the clubs have reached the finals. From the above visualization methodn we can see that there are 21 teams that have made it  the finals once and and 11 teams have made it to the finals 2-3 times and "Real Madrid" being the only team reaching the finals 16 times and winning 13 of them.

# %%
finalist_count_df['Wins'] = windf['Wins']

# %%
# Replace NaN values.
finalist_count_df['Wins'] = finalist_count_df['Wins'].fillna(0)
finalist_count_df['Wins'] = finalist_count_df['Wins'].astype(int)
finalist_count_df.head()

# %%
import plotly.express as px

fig = px.bar(finalist_count_df, x=finalist_count_df['finals'], y=finalist_count_df['Wins'],
             color='Team',
             height=400)
fig.show()

# %% [markdown]
# ##### In the previous visulization we have seen the number of times the teams have made it to the finals , but now we also compare them with the wins achieved in the finals by the teams.

# %%
y = []
years = req_df['Season']
for year in years:
  y.append(year[:4])

# %%
n = req_df.columns[0]
new_df = req_df.copy()
req_df.drop(n, axis=1, inplace=True)

# %%
req_df['Year'] = y

# %%
req_df.head()

# %%
req_stadium = req_df[['Year', 'Attendance']]
req_stadium.head()

# %%
winner_score_df = req_df[['Winner', 'Winner Score']]

# %%
win_score_group = winner_score_df.groupby(by=['Winner'], as_index=False)['Winner Score'].sum()

# %%
winner = win_score_group['Winner']
values = win_score_group['Winner Score']

windict = {}
for i in range(len(winner)):
  windict[winner[i]] = values[i]

# %%
finalist_score_df = req_df[['Finalist', 'Finalist Score']]

# %%
finalist_score_group = finalist_score_df.groupby(by=['Finalist'], as_index=False)['Finalist Score'].sum()

# %%
finalist = finalist_score_group['Finalist']
finalist_ = finalist_score_group['Finalist Score']

finalist_scores = {}
for i in range(len(finalist)):
  finalist_scores[finalist[i]] = finalist_[i]

# %%
# Total goals scored in the leaugue by a club
# windict
# finalist_scores
club_total_goals = {}
for k, v in finalist_scores.items():
  if k in windict.keys():
    club_total_goals[k] = v + windict[k]
  else:
    club_total_goals[k] = v

# %%
total_goals_by_clubs = pd.DataFrame(club_total_goals.items(), columns=['Team', 'Goals'])
total_goals_by_clubs.head()

# %%
req_df["Match Goals"] = req_df["Winner Score"] + req_df["Finalist Score"]

# %%
# Goal difference by match
req_df['Goal Difference'] = req_df['Winner Score'] - req_df['Finalist Score']
req_df.head()

# %%
# What was the goal margin?.
goal_diff_df = req_df[['Winner', 'Goal Difference']]
goal_diff_df

# %%
import plotly.express as px

fig = px.bar(goal_diff_df, x='Winner', y='Goal Difference')
fig.show()

# %% [markdown]
# #### The above visualization method shows the total number of winning goals scored by each clubs and the goal differences they won each time. By this we see that 'Real Madrid' have socred the most number of goals and segmentations shows the goal differences of each win.

# %% [markdown]
# #### Exploring the data - Now let's analyze and visulaize the data based on attendance to see the popularity of the each finals

# %%
import pandas as pd

championsleaguefinal_url = 'https://www.stadiumguide.com/figures-and-statistics/lists/champions-league-final-venues/'
dfs = pd.read_html(championsleaguefinal_url)
championsleaguefinalstadium_df = dfs[0]

# %%
championsleaguefinalstadium_df=championsleaguefinalstadium_df.sort_values(by=['Year'])
championsleaguefinalstadium_df=championsleaguefinalstadium_df.reset_index(drop=True)

# %%
import pandas as pd

stadium_url = 'https://www.stadiumguide.com/figures-and-statistics/lists/europes-largest-football-stadiums/'
dfs = pd.read_html(stadium_url)
df_stadium= dfs[0]

# %%
df_stadiumcapacity=championsleaguefinalstadium_df.merge(df_stadium, on='Stadium', how='left')

# %%
df_stadiumcapacity.drop(columns=['Unnamed: 0', 'Match','City','Country'], inplace=True)

# %%
df_stadiumcapacity=df_stadiumcapacity.drop(df_stadiumcapacity.tail(6).index)

# %%
df_final = pd.concat([og_df, df_stadiumcapacity], axis=1, sort=False)

# %%
df_final.drop(columns=['Season', 'Venue'], inplace=True)

# %%
df_final['Capacity']= df_final['Capacity'].fillna(df_final['Attendance'])

# %%
df_final['Capacity']=df_final['Capacity'].astype(int)

# %%
df_final.drop(df_final.tail(3).index,inplace=True) 

# %%
df_final['Year']=df_final['Year'].astype(int)

# %%
df_final.drop(df_final.head(1).index,inplace=True)

# %%
df_s = df_final
df_final['Exceeded Attendance Capacity'] = df_s["Attendance"] - df_s["Capacity"]

# %%
df_final = df_final[["Winner","Year","Exceeded Attendance Capacity"]]

# %%
df_final["Team with year"] = df_final["Winner"] + " : " + df_final["Year"].map(str)

# %%
df_final.drop(['Winner', 'Year'], axis=1, inplace = True)
df_final

# %%
import plotly.express as px
fig = px.bar(df_final, x='Team with year', y='Exceeded Attendance Capacity')
fig.show()

# %% [markdown]
# #### The above bar graph viz method shows the number of fans attended the matches and the stadium capacity.
# #### Not only the wins determine best club but the popularity of the clubs as well. We can visually see using bar graph that more fans have attended the 'Real Madird' final matches. In 1960 more number of fans have attended the 'Real Madrid' match and total of ~76K above the capacity of the stadium.

# %%
finalist_count = winx.copy()
for key, value in finx.items():
  if key in finalist_count.keys():
    finalist_count[key] = value + finalist_count[key]
  else:
    finalist_count[key] = value

# %%
finalist_count_df = pd.DataFrame(finalist_count.items(), columns=['Team', 'finals'])

# %%
finalist_count_df['Wins'] = windf['Wins']

# %%
# Replace NaN values.
finalist_count_df['Wins'] = finalist_count_df['Wins'].fillna(0)
finalist_count_df['Wins'] = finalist_count_df['Wins'].astype(int)
finalist_count_df

# %%
import seaborn as sns
from matplotlib import pyplot as plt

sns.violinplot(x = 'finals', y = 'Wins', data = finalist_count_df)
plt.xlabel("Number of match played")
plt.ylabel("Number of wins");

# %% [markdown]
# #### This was one of the failed experiment. We wanted to see the number of finals appeared and wins by each club but we couldn't interpret the results we wanted to.

# %%
#Kernel density estimation plot relating 'wins' and 'match_played' columns
sns.displot(
    data=finalist_count_df,
    x="Wins", hue="finals",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
    warn_singular=False,
)
plt.title("Kernel density estimation plot relating 'wins' and 'match_played' columns")
plt.xlabel("Number of Wins");

# %% [markdown]
# #### This was one of the failed experiment. We wanted to see the number of finals appeared and wins by each club but we couldn't interpret the results we wanted to.

# %% [markdown]
# #### So far we seen the number of wins, goals, and final appearances by each club. But the success of a team is dependent on how the performance of it's players. We will explore the total number of goals by players.

# %%
# Highest goal scorer in the competetion, Clean the data by filling the NaN values
top_player = og_df.copy()
top_player.fillna(0)

# %%
top_goal_win1_scorer = top_player['scorer 1'].value_counts()
top_goal_win2_scorer = top_player['scorer 2'].value_counts()
top_goal_win3_scorer = top_player['scorer 3'].value_counts()
top_goal_win4_scorer = top_player['scorer 4'].value_counts()
top_goal_win5_scorer = top_player['scorer 5'].value_counts()
top_goal_win6_scorer = top_player['scorer 6'].value_counts()
top_goal_win7_scorer = top_player['scorer 7'].value_counts()

# %%
import itertools 
import collections

# top_goal_win1_scorer.items(), top_goal_win2_scorer.items(), top_goal_win3_scorer.items(), top_goal_win4_scorer.items(), top_goal_win5_scorer.items(), top_goal_win6_scorer.items(), top_goal_win7_scorer.items()
def merge_dict(dict1, dict2):
    temp_dict = collections.defaultdict(int)
    for k, v in itertools.chain(dict1.items(), dict2.items()):
        temp_dict[k] += v
    return temp_dict

# %%
total_win_goals_by_player1 = merge_dict(top_goal_win1_scorer, top_goal_win2_scorer)
total_win_goals_by_player2 = merge_dict(total_win_goals_by_player1, top_goal_win3_scorer)
total_win_goals_by_player3 = merge_dict(total_win_goals_by_player2, top_goal_win4_scorer)
total_win_goals_by_player4 = merge_dict(total_win_goals_by_player3, top_goal_win5_scorer)
total_win_goals_by_player5 = merge_dict(total_win_goals_by_player4, top_goal_win6_scorer)
total_win_goals_by_player_final = merge_dict(total_win_goals_by_player5, top_goal_win7_scorer)
final_winning_goals_by_player = dict(total_win_goals_by_player_final)

# %%
# Opponents scores1
top_goal_los1_scorer = top_player['opponent scorer 1'].value_counts()
top_goal_los2_scorer = top_player['opponent scorer 2'].value_counts()
top_goal_los3_scorer = top_player['opponent scorer 3'].value_counts()

# %%
opp_goal_scorers1 = merge_dict(top_goal_los1_scorer, top_goal_los2_scorer)
final_opp_goals = merge_dict(opp_goal_scorers1, top_goal_los3_scorer)
final_opp_goals_by_player = dict(final_opp_goals)
total_goals_scored_by_player_overal = merge_dict(final_winning_goals_by_player, final_opp_goals_by_player)
total_goals_scored_by_player_overal = dict(total_goals_scored_by_player_overal)
total_goals_player_df = pd.DataFrame(total_goals_scored_by_player_overal.items(), columns=['Player', 'Goals'])

# %%
total_goals_player_df

# %%
players = ""
#print(total_goals_scored_by_player_overal)
for k, v in total_goals_scored_by_player_overal.items():
    players = players + (k + " ") * v
    #print(k)
print(players)

# %%
comment_words = ''

for val in players:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color='black', width=600, height=300, max_font_size=150, 
                      max_words=200).generate(players)
wordcloud.recolor(random_state=0)
plt.figure(figsize = (14, 10),facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# %% [markdown]
# #### The above figure represnets the word cloud of the players , the bigger the font of the player  name   the more mentions they have as the scorer in the finals of the Europa League. we can see that Stefano and Puskas , as they have the higest number of golas in the finals.

# %%
import seaborn as sns
plt.figure(figsize=(15,8))

sns.pairplot(req_df)

# %% [markdown]
# #### We also check the distribution of all the varaibles with each other usuing the seaborn library's pair pair plots .As the data is such we dont find any particular distribtion in the data that can be observed and stated . All the plots showe random distributions of the varaibles with each other.

# %%
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,8))
total_goals_player_df = total_goals_player_df.sort_values(by=['Goals'], ascending=False)
ax = sns.barplot(x="Player", y="Goals", data=total_goals_player_df)
ax.tick_params(axis='x', rotation=90)

# %% [markdown]
# #### The plot above histogram shows the players and the number of goals scored by them, Di stefano , Puskas and Ronaldo being the top three high scorers in Finals of Europa

# %%
req_df.head()

# %%
accu = req_df.groupby(['Winner']).sum().reset_index()

# %%
accu['Win_Accuracy'] = (accu['Winner ST'] / accu['Winner S']) * 100
accu['Fin_Accuracy'] = (accu['Finalist ST'] / accu['Finalist S']) * 100

# %%
accu.head(10)

# %%
accu['Winner'].tolist()

# %%
sorted_df = accu.sort_values(by=['Win_Accuracy'], ascending=True)
sorted_df_ten = sorted_df.head(10)

# %%

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,10)

cathegories = sorted_df_ten['Winner'].tolist()
percent = sorted_df_ten['Win_Accuracy'].tolist()
n = len(percent)
percent_circle = max(percent) / 100
r = 1.5
r_inner = 0.4
w = (r - r_inner) / n
colors = plt.cm.tab10.colors
fig, ax = plt.subplots()
ax.axis("equal")

for i in range(n):
    radius = r - i * w
    ax.pie([percent[i] / max(percent) * percent_circle], radius=radius, startangle=90,
           counterclock=False,
           colors=[colors[i]],
           labels=[f'{cathegories[i]} – {percent[i]}%'], labeldistance=None,
           wedgeprops={'width': w, 'edgecolor': 'white'},
          normalize=False)
    ax.text(0, radius - w / 2, f'{cathegories[i]} – {percent[i]}% ', ha='right', va='center')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### The above circular plot shows the Shot accuaray of the teams in the Finals , Ajax club has the lowest shot accuracy in the europa finals and Red Star belgrade the 2nd least accurate team with lowest shot accuray . we come across these new clubs as the matchs played by these teams is few  in number as compared to the big teams suvh as Madrid and Milan

# %%
sorted_df = accu.sort_values(by=['Win_Accuracy'], ascending=False)
sorted_df_ten = sorted_df.head(10)

# %%
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,10)


cathegories = sorted_df_ten['Winner'].tolist()
percent = sorted_df_ten['Win_Accuracy'].tolist()

n = len(percent)
percent_circle = max(percent) / 100

r = 1.5
r_inner = 0.4
w = (r - r_inner) / n

colors = plt.cm.tab10.colors
fig, ax = plt.subplots()
ax.axis("equal")

for i in range(n):
    radius = r - i * w
    ax.pie([percent[i] / max(percent) * percent_circle], radius=radius, startangle=90,
           counterclock=False,
           colors=[colors[i]],
           labels=[f'{cathegories[i]} – {percent[i]}%'], labeldistance=None,
           wedgeprops={'width': w, 'edgecolor': 'white'},
          normalize=False)
    ax.text(0, radius - w / 2, f'{cathegories[i]} – {percent[i]}% ', ha='right', va='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### The above circular plot shows the Shot accuaray of the teams in the Finals but from higer order , Aston villa club has the highest shot accuracy in the europa finals and Feyenooerd  the 2nd most accurate team with lowest shot accuray . we come across these new clubs as the matchs played by these teams is few in number as compared to the big teams suvh as Madrid and Milan here as well, we still dont find the two most dominating teams in the highest shot accuaray circle.

# %%



