from libraries import *
from params import *
from helper_funcs import *


def explode_qualifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the 'qualifiers' column in a DataFrame into multiple columns.
    
    :param df: The `df` parameter in the `explode_qualifiers_data` function is a pandas DataFrame that
    contains information about football events. The function processes the 'qualifiers' column in the
    DataFrame to extract and create new columns based on the data in the 'qualifiers' column.
    :type df: pd.DataFrame
    :return: The function `explode_qualifiers_data` returns a pandas DataFrame with the 'qualifiers'
    column exploded into multiple columns. The new columns are created based on the data in the
    'qualifiers' column
    """
    df = pd.concat([df, df['qualifiers'].apply(extract_qualifiers)], axis=1)
    return df

def events_feature_engineering(df : pd.DataFrame) -> pd.DataFrame:
    """
    The function `init_feature_engineering` calculates various features such as vertical length,
    distance, angle, pass stats, and goal stats for a given DataFrame.
    
    :param df: The code snippet you provided is a function init_feature_engineering that performs
    feature engineering on a DataFrame df
    - vertical length: Calculates the vertical length of a pass by subtracting the y-coordinate of the starting point from the y-coordinate of the ending point.
    - distance: Calculates the distance of a pass using the Euclidean distance formula.
    - angle: Calculates the angle of a pass using the arctan function.
    - isSuccessful: Converts the outcomeType column to a binary value, where 1 indicates a successful pass.
    - num_goals: Converts the eventType column to a binary value, where 1 indicates a goal event.
    """
    # vertical length
    df['x_length'] = df['x'] - df['endX']
    df['y_length'] = df['y'] - df['endY']

    # distance
    df['distance'] = np.round(np.sqrt(df['x_length']**2 + df['y_length']**2),3)

    # angle
    df['angle'] = np.round(np.arctan(df['y_length'] / df['x_length']),3)

    # pass stats
    df['isSuccessful'] = df['outcomeType'].apply(lambda x: int(x[-2]))

    # goal stats
    df['goal'] = df['eventType'].apply(lambda x: 1 if x == 'Goal' else 0)
    return df

def shots_feature_engineering(df : pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of shots, total shots, and proportion
    of shots for each team in a DataFrame.
    
    :param df: A pandas DataFrame containing shots event data. The DataFrame
    should have columns like 'isShot', 'jsonName', 'teamId', 'leagueSeason', and 'id'. The 'isShot'
    column should indicate whether a particular event is a shot or not
    :type df: pd.DataFrame
    :return: The function `shots_feature_engineering` takes a DataFrame as input, filters the rows where
    the column 'isShot' is True, groups the data by 'jsonName', 'teamId', and 'leagueSeason', counts the
    number of shots for each group, calculates the total number of shots for each 'jsonName', and then
    calculates the proportion of shots for each group. The function returns a
    """
    shots = (df
         .where(df['isShot'] == True)
         .groupby(['jsonName', 'teamId', 'leagueSeason']).agg({'id': 'count'})
         .reset_index()
         .rename({'id' : 'num_shots'}, axis=1))
    shots['total_shots'] = shots.groupby(['jsonName'])['num_shots'].transform('sum')
    shots['proportion_shots'] = np.round(shots['num_shots'] / shots['total_shots'], 2)
    return shots

def goals_feature_engineering(df):
    """
    Calculates the number of goals, total goals, and proportion
    of goals for each team in a DataFrame.

    :param df: A pandas DataFrame containing goal event data. The DataFrame
    should have columns 'eventType', 'jsonName', 'teamId', 'leagueSeason', and 'id'. The 'eventType'
    column should indicate whether a particular event is a goal or not
    :type df: pd.DataFrame
    :return: The function `goals_feature_engineering` takes a DataFrame as input, filters the rows where
    the column 'eventType' is 'Goal', groups the data by 'jsonName', 'teamId', and 'leagueSeason', counts
    the number of goals for each group, calculates the total number of goals for each 'jsonName', and then
    calculates the proportion of goals for each group. The function returns a DataFrame with additional
    columns 'num_goals', 'total_goals', 'proportion_goals', 'outcome', 'is_win', 'is_draw', and 'is_loss'.
    """
    goals = (df
         .where(df['eventType'] == 'Goal')
         .groupby(['jsonName', 'teamId', 'leagueSeason']).agg({'id': 'count'})
         .reset_index()
         .rename({'id' : 'num_goals'}, axis=1))
    goals['total_goals'] = goals.groupby(['jsonName'])['num_goals'].transform('sum')
    goals['proportion_goals'] = np.round(goals['num_goals'] / goals['total_goals'], 2)
    goals['outcome'] = np.where(goals['num_goals'] > goals['total_goals'] / 2, 2, np.where(goals['num_goals'] == goals['total_goals'] / 2, 1, 0))
    goals['is_win'] = (goals['outcome'] == 3).astype(int)
    goals['is_draw'] = (goals['outcome'] == 1).astype(int)
    goals['is_loss'] = (goals['outcome'] == 0).astype(int)
    return goals

def passes_feature_engineering(df : pd.DataFrame) -> pd.DataFrame:
    passes = df[df['eventType'] == 'Pass']
    passes['Zone'] = LabelEncoder().fit_transform(passes['Zone'])
    passes['Length'] = passes['Length'].astype(float)

    passes = (passes
            .groupby(['jsonName', 'teamId', 'leagueSeason'])
            .agg(num_passes=('isSuccessful', 'count'),
                num_successful_passes=('isSuccessful', 'sum'),
                num_long_balls=('Longball', 'sum'),
                num_throw_ins=('ThrowIn', 'sum'),
                num_crosses=('Cross', 'sum'),
                num_corners=('CornerTaken', 'sum'),
                num_through=('Throughball', 'sum'),
                num_chips=('Chipped', 'sum'),
                num_headed=('HeadPass', 'sum'),
                num_key_passes=('KeyPass', 'sum'),
                num_counters=('FastBreak', 'sum'),
                pass_distance=('distance', 'mean'),
                pass_angle=('angle', 'mean'),
                pass_length=('Length', 'mean'),
                zone=('Zone', 'mean'),
                x_proxy_length=('x_length', 'mean'),
                y_proxy_length=('y_length', 'mean'),
                )
            .reset_index())
    passes['pass_success_proportion'] = np.round(passes['num_successful_passes'] / passes['num_passes'], 2)
    return passes
