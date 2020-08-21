# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:03:20 2020

@author: Sam
"""

import keyring
import requests
import pandas as pd
import numpy as np
import bs4 as bs
from io import StringIO 
import math
from datetime import date

pd.set_option("display.max_columns",10)
#%% File Grabber

def get_files(season):

    login_url = "https://kenpom.com/handlers/login_handler.php"
    
    username = keyring.get_password("kenpom","username")
    getin = keyring.get_password("kenpom","getin")
    
    payload = {
            "email" : username,
            "password": getin,
            "submit": "Login!"
            }
    
    session_requests = requests.Session()
    
    session_requests.post(
            login_url,
            data = payload
            )
    
    season = str(season)
    
    file_url = "https://kenpom.com/getdata.php?file=summary"
    
    url_to_get = file_url+season[2:4]
    
    result = session_requests.get(url_to_get,allow_redirects = False)
    
    response_content = result.content.decode("utf-8")
    
    cont_to_str = StringIO(response_content)
    
    summary = pd.read_csv(cont_to_str, sep =",") 
    
    return(summary)
    
#%% D1 Stats
    
def d1_stats(season):
   
    login_url = "https://kenpom.com/handlers/login_handler.php"
    
    username = keyring.get_password("kenpom","username")
    getin = keyring.get_password("kenpom","getin")
    
    payload = {
            "email" : username,
            "password": getin,
            "submit": "Login!"
            }
    
    session_requests = requests.Session()
    
    session_requests.post(
            login_url,
            data = payload
            )
    
    season = str(season)
    
    file_url = "https://kenpom.com/team.php?team=Vanderbilt&y="
    
    url_to_get = file_url+season
    
    result = session_requests.get(url_to_get,allow_redirects = False)
    
    tabs = pd.read_html(result.content)
    
    d1_avg = tabs[0]
    
    d1_avg.columns = ["Category","Offense","Defense","D1_Avg"]
    
    d1_avg.iat[0,0] = "Efficiency"
    d1_avg.iat[1,0] = "Tempo"
    
    d1_avg = d1_avg[d1_avg["D1_Avg"].notna()]
    d1_avg = d1_avg[["Category","D1_Avg"]]
    d1_avg = d1_avg.assign(Category = lambda x: x['Category'].str.replace(":",""))
    
    return(d1_avg)

#%% Game Numberer
    
def game_numberer(teams):
   
    number_of_rounds = range(1,int(math.log(teams, 2))+1)
    
    round_list = 2**(np.array(number_of_rounds)-1)
    round_list = np.sort(round_list)[::-1]
    
    all_games = []
    starter = 0
    for i in round_list:
        rep_times = teams/i
        game_numbers = np.array(range(starter+1,starter+i+1))
        game_numbers = np.repeat(game_numbers,rep_times)
        all_games.append(game_numbers)
        starter = game_numbers[-1]
        
    df = pd.DataFrame(all_games).transpose()
    
    col_names = []
    for i in round_list:
        col_names.append("R"+str(i))
        
    df.columns = col_names
    
    return(df)
    
#%% Team Codes
    
def team_codes_bart():
    bart = requests.get("https://barttorvik.com/teammatch.json")
    
    team_codes = pd.DataFrame.from_dict(bart.json(),orient = "index")
    
    team_codes.index.name = 'Name'
    team_codes.reset_index(inplace = True)
    team_codes.columns = ['Name','Team']
    
    return(team_codes)

#%% Bracket Matrix
    
def bracket_matrix():
    
    try:
        team_codes
    except NameError:
        team_codes_bart()
    
    url = "http://www.bracketmatrix.com/"
    
    mat = pd.read_html(url)[0]
    mat = mat.iloc[:,0:2]
    mat = mat[mat[0].notna()]
    mat.columns = ["Seed","Name"]
    
    regions_by_number = pd.DataFrame(
            {'RegionNo': [1,2,3,4],
            'Region': ["MIDWEST","EAST","WEST","SOUTH"]}
            )
    
    seeds = (mat
     .astype({'Seed':'int'})
     .astype({'Seed':'str'})
     .assign(RegionNo = lambda x: x.groupby(['Seed']).cumcount())
     .assign(RegionNo = lambda x: x['RegionNo']+1)
     .query('RegionNo <= 4')
     .join(regions_by_number.set_index('RegionNo'),on = 'RegionNo')
     .assign(Seed = lambda x: pd.Categorical(x['Seed'],np.array([1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]).astype('str').tolist(),
                                             ordered=True))
     .sort_values(by=['RegionNo','Seed'])
     .join(team_codes.set_index('Name'),on = 'Name')
     )
     
    return(seeds[['Region','Seed','Team']])
    
#%% Bracket Maker
    
def bracket_maker(season,bracket_type,no_of_teams):
    
    season = str(season)
    bracket_type = "regular"
    
    try:
        team_codes
    except NameError:
        team_codes_bart()
        
    url = "https://www.ncaa.com/brackets/basketball-men/d1"+"/"+season
    
    page = bs.BeautifulSoup(requests.get(url).content)
        
    regions = [i.text for i in page.select(".region-left h3")]
    regions.extend([i.text for i in page.select(".region-right h3")])
    regions = np.repeat(regions,repeats = 16)
    
    seeds = [i.text for i in page.select(".region-left .round-1 .seed")]
    seeds.extend([i.text for i in page.select(".region-right .round-1 .seed")])
    
    teams = [i.text for i in page.select(".region-left .round-1 .name")]
    teams.extend([i.text for i in page.select(".region-right .round-1 .name")])
    
    r32_pods = np.repeat(range(1,33),2).tolist()
    
    bracket = pd.DataFrame(list(zip(regions,seeds,teams,r32_pods)),
                 columns = ['Region','Seed','Team',"R32_Pod"])
    
    bracket = (bracket
     .astype({'Seed':'int'})
     .assign(Seed1 = lambda x: x.groupby(['R32_Pod']).transform('sum')['Seed'])
     .assign(Seed1 = lambda x: 17 - x['Seed1'])
     .astype({'Seed1':'int'})
     .assign(Seed = lambda x: np.select([x.Seed.isnull().values.any()],[x.Seed1],default = x.Seed))
     .loc[:,['Region','Seed','Team']]
     .rename(columns = {'Team':'Name'})
     .join(team_codes.set_index('Name'),on = 'Name')
     .loc[:,['Region','Seed','Team']]
     )
    
    ## Get Files
        
    summary = get_files(season = season)
    
    ## Find Leagues
    
    league_url = "https://kenpom.com/index.php?y="+str(season)
    
    league_soup = pd.read_html(requests.get(league_url).content)[0].iloc[:,1:3]
    league_soup.columns = ['Name','League']
    
    high_majors = ["ACC","B10","SEC","B12","P12","BE"]
    
    hm_mm = (league_soup
     .assign(League= lambda x: np.where(x.League.isin(high_majors),1,0))
     .assign(Name = lambda x: x['Name'].str.replace(" \\d+$",""))
     .set_index('Name')
     .join(team_codes.set_index('Name'),on = 'Name')
     .loc[:,['Team','League']]
     )
    
    team_stats = (summary
     .rename(columns = {'TeamName':'Team'})
     .loc[:,['Team','AdjEM','AdjTempo']]
     )
    
    full_bracket = (bracket
     .join(hm_mm.set_index('Team'),on = 'Team')
     .join(team_stats.set_index('Team'),on = 'Team')
     )
    
    if date.today().year!=season or bracket_type=="projected":
        bracket_return = pd.concat([full_bracket,game_numberer(no_of_teams)],axis = 1)
        return(bracket_return)
        
    ## Play-Ins
        
    playin_teams = [i.text for i in page.select(".game .name")]
    
    playin_regions = [i.text for i in page.select(".game h4")]
    playin_regions = np.repeat(playin_regions,repeats = 2)
    
    playin_seeds = [i.text for i in page.select(".game .seed")]
    
    try:
        tempo
    except NameError:
        tempo = (d1_stats(season = season)
                 .astype({'D1_Avg':'float'})
                 .query('Category=="Tempo"')
                 .iat[0,1]
                )
    
    playins = (pd.DataFrame(list(zip(playin_teams,playin_regions,playin_seeds)),columns = ['Name','Region','Seed'])
     .join(team_codes.set_index('Name'),on = 'Name')
     .join(
           (summary
            .rename(columns = {'TeamName':'Name'})
            .join(team_codes.set_index('Name'),on = 'Name')
            .loc[:,['Team','AdjEM','AdjTempo']]
            .set_index('Team')
            ),
           on = 'Team'
           )
     .join(hm_mm.set_index('Team'),on = 'Team')
     .assign(Game = lambda x: x.groupby(['Region','Seed']).grouper.group_info[0])
     )
     
    playins = (playins
     .join(
           (playins
            .rename(columns = {'Team':'Opponent','League':'OpponentLeague','AdjEM':'OpponentEM','AdjTempo':'OpponentTempo'})
            .loc[:,['Opponent','OpponentLeague','OpponentEM','OpponentTempo','Game']]
            .set_index('Game')
            ),
           on = 'Game'
           )
     .query('Team!=Opponent')
     .assign(
             EM_Margin = lambda x: x.AdjEM - x.OpponentEM,
             Game_Tempo = lambda x: x.AdjTempo + x.OpponentTempo - tempo
             )
     .assign(EM_Margin = lambda x: np.select([np.logical_and(x.League==1,x.OpponentLeague==0)],
                                              [x.EM_Margin+1.8],
                                              default = x.EM_Margin))
     .assign(EM_Margin = lambda x: np.select([np.logical_and(x.League==0,x.OpponentLeague==1)],
                                              [x.EM_Margin-1.8],
                                              default = x.EM_Margin))
     .assign(MOV = lambda x: x.EM_Margin*x.Game_Tempo/100)
     .assign(Pct = lambda x: 1/(1+np.exp((-.146)*x.MOV)))
     )
     
    playin_odds = (playins
     .groupby(['Region','Seed']).apply(lambda x: (x['League']*x['Pct']).sum())
     .reset_index()
     .rename(columns = {0:'League1'})
     .merge(
           (playins
            .groupby(['Region','Seed']).apply(lambda x: (x['AdjEM']*x['Pct']).sum())
            .reset_index()
            .rename(columns = {0:'AdjEM1'})
            ),
            on = ['Region','Seed']
           )
     .merge(
           (playins
            .groupby(['Region','Seed']).apply(lambda x: (x['AdjTempo']*x['Pct']).sum())
            .reset_index()
            .rename(columns = {0:'AdjTempo1'})
            ),
            on = ['Region','Seed']
           )
     .assign(Team1 = ["Play-In 1","Play-In 2","Play-In 3","Play-In 4"])
     )
           
    full_bracket = (full_bracket
     .astype({'Seed':'str'})
     .merge(playin_odds,on = ['Region','Seed'],how = "left")
     .assign(Team = lambda x: np.select([x.Team.isnull().values.any()],[x.Team1],default = x.Team))
     .assign(League = lambda x: np.select([x.League.isnull().values.any()],[x.League1],default = x.League))
     .assign(AdjEM = lambda x: np.select([x.AdjEM.isnull().values.any()],[x.AdjEM1],default = x.AdjEM))
     .assign(AdjTempo = lambda x: np.select([x.AdjTempo.isnull().values.any()],[x.AdjTempo1],default = x.AdjTempo))
     .loc[:,['Region','Seed','Team','League','AdjEM','AdjTempo']]
     )
    
    bracket_return = pd.concat([full_bracket,game_numberer(no_of_teams)],axis = 1)
    
    return(bracket_return)

#%% Matchup Grid
    
def matchup_grid(dataframe):

    no_of_teams = len(dataframe.index)
    round_list = ["R"+str(i) for i in 2**np.array(range(0,int(math.log(no_of_teams,2))))[::-1]]
        
    cuts = np.flip(no_of_teams-2**np.array(range(0,int(math.log(no_of_teams,2))+1))+1)
    games = range(1,no_of_teams,1)
    
    games_by_round = pd.cut(games, bins = cuts,labels = round_list,right = False)
    
    binders = []
    
    for i in games:
    
        round = games_by_round[i-1]
        prior_round = round_list[round_list.index(round)-1]
        
        subdf1 = (dataframe
         .loc[:,['Team']+round_list]
         )
        
        subdf2 = subdf1[subdf1.eq(i).any(1)]
        
        matchup_df = pd.DataFrame([(x, y) for x in subdf2['Team'] for y in subdf2['Team']])
        
        if round==round_list[0]:
            
            binder = (matchup_df
             .rename(columns = {0:"Team1",1:"Team2"})
             .query('Team1 != Team2')
             .assign(Game = i)
             )
            
        else:
        
            path1 = (dataframe
             .loc[:,['Team',prior_round]]
            )
            
            path2 = (dataframe
             .loc[:,['Team',prior_round]]
            )
            
            path1.columns = ['Team1','Path1']
            path2.columns = ['Team2','Path2']
            
            binder = (matchup_df
             .rename(columns = {0:"Team1",1:"Team2"})
             .join(path1.set_index('Team1'),on = "Team1")
             .join(path2.set_index('Team2'),on = "Team2")
             .query('Path1 != Path2')
             .assign(Game = i)
             .loc[:,['Game','Team1','Team2']]
             )
        
        binders.append(binder)
        
    return(pd.concat(binders))
    
#%% Odds to Win

def odds_to_win(dataframe,bracket):
    
    try:
        tempo
    except NameError:
        tempo = float(
        (d1_stats(season = season)
         .query('Category=="Tempo"')
         .iat[0,1]
        )
    )
    
    cols_to_select = ['Team','AdjEM','AdjTempo','League']
    cols1 = [i+'1' for i in cols_to_select]
    cols2 = [i+'2' for i in cols_to_select]
    
    o1 = bracket.loc[:,cols_to_select]
    o1.columns = cols1
    
    o2 = bracket.loc[:,cols_to_select]
    o2.columns = cols2
    
    return_df = (dataframe
     .join(o1.set_index('Team1'), on = 'Team1')
     .join(o2.set_index('Team2'), on = 'Team2')
     .assign(EM = lambda x: x.AdjEM1 - x.AdjEM2,
             Game_Tempo = lambda x: x.AdjTempo1 + x.AdjTempo2 - tempo)
     .assign(MOV = lambda x: x.EM*x.Game_Tempo/100+1.8*x.League1-1.8*x.League2)
     .assign(Pct = lambda x: 1/(1+np.exp(-.146*x.MOV)))
     .loc[:,['Game','Team1','Team2','Pct']]
     )
     
    return(return_df)

#%% log5
    
def log5(dataframe,bracket,return_type = "log5"):

    all_rounds = (dataframe
     .loc[dataframe.Round==round_list[0]]
     .loc[:,['Team1','Round','Pct']]
     .rename(columns = {'Team1':'Team'})
     )
    
    all_round_matchups = pd.DataFrame()
    
    for i in round_list[1:]:
        
        prior_round = round_list[round_list.index(i)-1]
        
        #if i==round_list[1]:
        #    prev_round = all_rounds
        #else:
            
        prev_round = all_rounds.loc[all_rounds.Round==prior_round]
        
        new_round = (dataframe
         .loc[dataframe.Round==i]
         .assign(Previous_Round = prior_round)
         .merge(prev_round.rename(
                 columns = {'Round':'Previous_Round','Pct':'Team2_Odds_To_Be_There'}
                 ),
               left_on = ['Previous_Round','Team2'],
               right_on = ['Previous_Round','Team'],
               how = "left"
               )
         .merge(prev_round.rename(
                 columns = {'Round':'Previous_Round','Pct':'Team1_Odds_To_Be_There'}
                 ),
               left_on = ['Previous_Round','Team1'],
               right_on = ['Previous_Round','Team'],
               how = "left"
               )
         )
        
        ## This is where you would theoretically put the matchups code
        
        new_round_matchups = (new_round
         .assign(Odds_To_Occur = lambda x: x.Team1_Odds_To_Be_There*x.Team2_Odds_To_Be_There)
         .loc[:,['Team1','Team2','Round','Pct','Odds_To_Occur']]
         )
        
        if i==round_list[1]:
            
            joiner1 = (bracket
             .melt(id_vars = 'Team',value_vars = round_list[0],var_name = "Round",value_name = "Game")
             .rename(columns = {'Team':'Team1'})
             )
            
            joiner2 = (joiner1
             .merge(joiner1.rename(columns = {'Team1':'Team2'}),
                    left_on = ['Round','Game'],
                    right_on = ['Round','Game'],
                    how = "left"
                     )
             .query('Team1 != Team2')
             .loc[:,['Game','Team1','Team2']]
             )
             
            round1 = (all_rounds
             .merge(joiner2,
                    left_on = 'Team',
                    right_on = 'Team1',
                    how = "inner")
             .loc[:,['Team1','Team2','Round','Pct']]
             .assign(Odds_To_Occur = 1)
             )
              
            all_round_matchups = all_round_matchups.append(round1)
              
        all_round_matchups = all_round_matchups.append(new_round_matchups)
        
        ## Matchups code end
        
        new_round = (new_round
         .groupby(['Team1','Round']).apply(lambda x: (x['Team1_Odds_To_Be_There']*x['Team2_Odds_To_Be_There']*x['Pct']).sum())
         .reset_index()
         .rename(columns = {'Team1':'Team',0:'Pct'})
         )
        
        all_rounds = all_rounds.append(new_round)
        
    if return_type == "matchups":
        return(all_round_matchups)
        
    return_df = (all_rounds
     .pivot_table(index = 'Team',columns = 'Round',values = 'Pct')
     .reset_index()
     .loc[:,['Team']+round_list]
     )
    
    return(return_df)

#%% Upsets
    
def upsets(dataframe,bracket):

    upset_pxs = (log5(dataframe = dataframe,bracket = bracket,return_type = "matchups")
     .join((bracket
            .rename(columns = {'Team':'Team1','Seed':'Seed1'})
            .loc[:,['Team1','Seed1']]
            .set_index('Team1')
            ),
            on = 'Team1'
           )
     .join((bracket
            .rename(columns = {'Team':'Team2','Seed':'Seed2'})
            .loc[:,['Team2','Seed2']]
            .set_index('Team2')
            ),
            on = 'Team2'
           )
     .merge((bracket
              .melt(id_vars = 'Team',value_vars = round_list,var_name = "Round",value_name = "Game")
              .rename(columns = {'Team':'Team1'})
             ),
            left_on = ['Team1','Round'],
            right_on = ['Team1','Round'],
            how = "left"
            )
     .assign(Seed_Diff = lambda x: x.Seed1-x.Seed2)
     .assign(Odds = lambda x: x.Odds_To_Occur*x.Pct)
     .assign(Index = lambda x: range(1,len(x.index)+1))
    )
        
    upset_probs = []
        
    for i in range(1,len(upset_pxs.index)+1):
    
        index_row = upset_pxs.loc[upset_pxs.Index==i]
        team1 = index_row.iloc[0]['Team1']
        team2 = index_row.iloc[0]['Team2']
        game = index_row.iloc[0]['Game']
        
        cant_happen_indices = upset_pxs.loc[(upset_pxs['Game']<game) & (upset_pxs['Team2'].isin([team1,team2]))]['Index']
        
        seed1 = index_row.iloc[0]['Seed1']
        seed_diff = index_row.iloc[0]['Seed_Diff']
        win_odds = index_row.iloc[0]['Odds']
        
        if seed_diff <= 0:
            odds_of_no_higher_seeded_upsets = 0
        else:
            o1 = upset_pxs.loc[(~upset_pxs['Index'].isin(cant_happen_indices)) & (upset_pxs['Index'] != i)]
            o2 = o1.loc[(o1['Seed_Diff'] > seed_diff) | ((o1['Seed_Diff']==seed_diff) & (o1['Seed1'] > seed1))]
            odds_of_no_higher_seeded_upsets = o2.assign(Inverse = lambda x: 1 - x['Odds'])['Inverse'].prod()
        
        tiebreak_odds = upset_pxs.loc[(~upset_pxs['Index'].isin(cant_happen_indices)) & 
                                      (upset_pxs['Seed_Diff']==seed_diff) & 
                                      (upset_pxs['Seed1']==seed1)]
        
        if len(tiebreak_odds.index) > 0:
            long_string_object = tiebreak_odds[tiebreak_odds['Index']==i]['Odds']/tiebreak_odds['Odds'].sum()
            long_string_object = long_string_object.values
            if np.isnan(long_string_object):
                long_string_object = 0
                
        else: 
            long_string_object = 0
            
        if i%50==0:
            print(i)
            
        upset_probs.append(long_string_object*odds_of_no_higher_seeded_upsets*win_odds)
        
    biggest_upsets = (upset_pxs
     .assign(Odds_Best = upset_probs/sum(upset_probs))
     .groupby('Team1')['Odds_Best'].sum()
     .reset_index()
     )
    
    return(biggest_upsets)
    
#%% R16-R8 Helper
    
def r16_r8_helper(dataframe,bracket):
    
    base_df = (bracket
     .loc[:,['Region','Seed','Team']]
     .join(dataframe.set_index('Team'), on = 'Team')
     )
        
    r16_low_seeds_by_pod = (base_df
     .assign(R16_Pod = np.repeat(range(1,17),repeats = 4))
     .query('Seed > 4')
     .groupby('R16_Pod')['R16'].sum()
     .reset_index()
     .rename(columns = {'R16':'R16_Low_Seeds'})
     .assign(Other_R16_Low_Seeds = lambda x: sum(x.R16_Low_Seeds)-x.R16_Low_Seeds)
     .loc[:,['R16_Pod','Other_R16_Low_Seeds']]
    )
    
    r8_low_seeds_by_pod = (base_df
     .assign(R8_Pod = np.repeat(range(1,9),repeats = 8))
     .query('Seed > 2')
     .groupby('R8_Pod')['R8'].sum()
     .reset_index()
     .rename(columns = {'R8':'R8_Low_Seeds'})
     .assign(Other_R8_Low_Seeds = lambda x: sum(x.R8_Low_Seeds)-x.R8_Low_Seeds)
     .loc[:,['R8_Pod','Other_R8_Low_Seeds']]
    )
    
    bdf2 = (base_df
     .assign(R16_Pod = np.repeat(range(1,17),repeats = 4))
     .assign(R8_Pod = np.repeat(range(1,9),repeats = 8))
     .join(r16_low_seeds_by_pod.set_index('R16_Pod'), on = 'R16_Pod')
     .join(r8_low_seeds_by_pod.set_index('R8_Pod'), on = 'R8_Pod')
     .assign(Index = lambda x: range(1,len(x.index)+1))
     )
    
    bdf3 = pd.DataFrame()
    
    for i in range(1,len(bdf2.index)+1):
    
        index_row = bdf2.loc[bdf2['Index']==i]
        
        pod = index_row.iloc[0]['R16_Pod']
        team = index_row.iloc[0]['Team']
        odds = index_row.iloc[0]['R16']
        
        bdf4 = (bdf2
         .assign(R16 = lambda x: np.where(x.R16_Pod==pod,0,x.R16))
         .assign(R16 = lambda x: np.where(x.Team==team,1,x.R16))
         .assign(R16_Value = lambda x: np.where(x.Seed > 4, x.R16*.28/sum(x.R16[x.Seed>4]),0))
         .assign(Scenario = team, Odds = odds)
         )
        
        bdf3 = bdf3.append(bdf4)
    
    r16 = (bdf3
    .groupby('Team').apply(lambda x: sum(x['R16_Value']*x['Odds'])/16)
    .reset_index()
    .rename(columns = {0:'R16_Value'})
    )
    
    bdf5 = pd.DataFrame()
    
    for i in range(1,len(bdf2.index)+1):
    
        index_row = bdf2.loc[bdf2['Index']==i]
        
        pod = index_row.iloc[0]['R8_Pod']
        team = index_row.iloc[0]['Team']
        odds = index_row.iloc[0]['R8']
        
        bdf6 = (bdf2
         .assign(R8 = lambda x: np.where(x.R8_Pod==pod,0,x.R8))
         .assign(R8 = lambda x: np.where(x.Team==team,1,x.R8))
         .assign(R8_Value = lambda x: np.where(x.Seed > 2, x.R8*.10/sum(x.R8[x.Seed>2]),0))
         .assign(Scenario = team, Odds = odds)
         )
        
        bdf5 = bdf5.append(bdf6)
        
    r8 = (bdf5
    .groupby('Team').apply(lambda x: sum(x['R8_Value']*x['Odds'])/8)
    .reset_index()
    .rename(columns = {0:'R8_Value'})
    )
    
    df = (base_df
     .join(r8.set_index('Team'), on = 'Team')
     .join(r16.set_index('Team'), on = 'Team')
     .loc[:,['Region','Seed','Team','R32','R16_Value','R8_Value','R4','R2','R1']]
     .rename(columns = {'R16_Value':'R16','R8_Value':'R8'})
     )
    
    return(df)
    
#%% EV
    
def ev_calc(dataframe,bracket,upset_df):
    
    ev = (r16_r8_helper(dataframe,bracket)
     .assign(R1 = lambda x: .1*x.R1,
             R2 = lambda x: .09*x.R2,
             R4 = lambda x: .055*x.R4,
             R32_Loss = lambda x: (1-x.R32)/32)
     .assign(R32 = 0,
             R32_Loss = lambda x: .04*x.R32_Loss)
     .join((upset_df
           .rename(columns = {'Odds_Best':'Biggest_Upset','Team1':'Team'})
           .assign(Biggest_Upset = lambda x: .05*x.Biggest_Upset)
           .set_index('Team')
           ),
           on = 'Team'
           )
     .assign(Total = lambda x: x[['R32','R16','R8','R4','R2','R1','R32_Loss','Biggest_Upset']].sum(axis = 1))
     )
     
    return(ev)
