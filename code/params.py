# directory where json data is stored
data_dir = '/Users/rustya/thesis/data/WhoScored/events/*'

merge_key = ['jsonName', 'teamId', 'leagueSeason']

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