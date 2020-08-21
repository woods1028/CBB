# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:29:44 2020

@author: Sam
"""

#%% Gather Items
 
season = 2019 
no_of_teams = 64

round_list = ["R"+str(i) for i in 2**np.array(range(0,int(math.log(no_of_teams,2))))[::-1]]

team_codes = team_codes_bart()

tempo = float(
        (d1_stats(season = season)
         .query('Category=="Tempo"')
         .iat[0,1]
        )
    )
        
bracket = bracket_maker(season = 2019,bracket_type = "regular")

games_by_round_df = (bracket
 .loc[:,['Team']+round_list]
 .melt(id_vars = ['Team'],value_vars = round_list,
       var_name = 'Round',value_name = 'Game')
 .loc[:,['Round','Game']]
 .drop_duplicates()
 )

all_matchups_w_round = (odds_to_win(matchup_grid(bracket))
 .join(games_by_round_df.set_index('Game'), on = 'Game')
)

log5_df = log5(dataframe = all_matchups_w_round,bracket = bracket,return_type = "log5")

dataframe = all_matchups_w_round

biggest_upsets = upsets(dataframe = all_matchups_w_round,bracket = bracket)

ev = ev_calc(dataframe = log5_df, bracket = bracket, upset_df = biggest_upsets)


