from libraries import *

# directory where json data is stored
data_dir = "C:/Users/rustya/Desktop/mds-thesis/data/WhoScored/events/*"

merge_key = ['jsonName', 'teamId', 'leagueSeason']

# columns that should not be filled with 0 because they are coordinates
cols_not_to_fill = ['x', 'y', 'endX', 'endY', 'PassEndX', 'PassEndY', 'Angle', 'angle', 'Length', 'PassLength', 'y_length', 'x_length', 'distance', 'Distance']

redundant_cols = ['id', 
                  'eventId',
                  'isTouch',
                  'outcomeType',
                  'satisfiedEventsTypes',
                  'type',
                  'playerId',
                  'relatedEventId',
                  'relatedPlayerId',
                  ]

aed_leakage = ['jsonName', 'winning_team', 'outcome_1', 'outcome_2', 'num_goals_1', 'num_goals_2', "total_goals_1",	'total_goals_2', 'proportion_goals_1', 'proportion_goals_2', 'teamId_1', 'teamId_2']
naed_leakage = ['jsonName', 'teamId', 'leagueSeason', 'num_goals', 'total_goals', 'proportion_goals', 'outcome', 'is_win', 'is_draw', 'is_loss', 'angle']
ncaed_leakage = ['jsonName', 'teamId', 'idx', 'leagueSeason', 'num_goals', 'total_goals', 'proportion_goals', 'outcome', 'is_win', 'is_draw', 'is_loss']

modelling_dict = {
    'aed' : {'leakage' : aed_leakage, 
             'target' : 'winning_team'},
    'naed' : {'leakage' : naed_leakage,
              'target' : 'outcome'},
    'ncaed' : {'leakage' : ncaed_leakage,
               'target' : 'outcome'}
}