from libraries import *
from params import *

def read_footy_json(js : json) -> pd.DataFrame:
    """
    Reads a JSON file containing football events data and returns a
    pandas DataFrame with additional columns for the JSON file name and league season.
    
    :param js: The `js` parameter in the `read_footy_json` function is expected to be a JSON object that
    contains information about football events. The function reads this JSON object, extracts the
    'events' data, and creates a pandas DataFrame with additional columns 'jsonName' and 'leagueSeason'
    based
    :type js: json
    :return: A pandas DataFrame containing the data from the 'events' key of the JSON file, with
    additional columns 'jsonName' and 'leagueSeason' extracted from the file name.
    """
    df = pd.DataFrame(json.load(js)['events'])
    df['jsonName'] = js.name.split('/')[-1]
    df['leagueSeason'] = js.name.split('/')[-2]
    return df

def pretify_json_data(directory: str) -> 'csv':
    """
    Reads JSON files in batches, concatenates them into a DataFrame,
    and writes the data to CSV files with a specific naming convention.
    
    :param directory: The `directory` parameter in the `pretify_json_data` function is a string that
    represents the path to a directory containing JSON files that you want to process and convert into a
    CSV format
    :type directory: str
    :return: returns the string 'Compilation completed'.
    """
    filenames = []
    for subfloder in glob(directory):
        filenames.extend(glob(f'{subfloder}/*'))
    # reading files by batches of 200, write to parquet, and then read again. 
    # add leagueSeason to the name for easy filtering
    c = 0
    for i in range(0, len(filenames), 200):
        print(f'Processing files {i} to {i+200}')
        df = pd.DataFrame()
        for filename in filenames[i:i+200]:
            with open(filename) as f:
                df = pd.concat([df, read_footy_json(f)])
        df.to_parquet(f'epl_{c}.parquet')
        c += 1
    return 'Compilation completed'

def load_prettified_data() -> pd.DataFrame:
    """
    Reads data from multiple .parquet files with footy data into a pandas DataFrame,
    converting the 'qualifiers' column values to Python literals.
    :return: A pandas DataFrame containing data loaded from multiple parquet files named 'epl_0.parquet',
    'epl_1.parquet', 'epl_2.parquet', 'epl_3.parquet', 'epl_4.parquet', and 'epl_5.parquet'
    """
    df = pd.DataFrame()
    for i in range(6):
        df = pd.concat([df, pd.read_parquet(f'data/epl_{i}.parquet')])
    return df

def extract_qualifiers(row: "pd.Series"):
    """
    This function extracts data from a pandas Series object and creates a new Series with display names
    as index and corresponding values.
    
    :param row: A pandas Series containing data in a specific format. The function `extract_data` is
    designed to extract and process information from this Series
    :type row: "pd.Series"
    :returns: a pandas Series with display names as index and corresponding values.
    """

    display_names = []
    values = []
    for item in row:
        try:
            display_names.append(item['type']['displayName'])
            if 'value' in item and item['value'] is not None:
                values.append(item['value'])
            else:
                values.append(1 if 'value' in item['type'] else item['type']['value'])
        except:
            pass
    return pd.Series(values, index=display_names)

def split_qualifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the qualifiers in a DataFrame column into individual dictionaries and then
    converts them into a single dictionary.
    
    :param df: The function `split_qualifiers` takes a pandas DataFrame `df` as input. The DataFrame is
    expected to have a column named 'qualifiers' which contains a list of dictionaries. The function
    splits the qualifiers into individual dictionaries, converts them into a specific format, and then
    returns the modified DataFrame
    :type df: pd.DataFrame
    :return: returns the modified DataFrame.
    """
    # split qualifiers
    df['splitted_qualifiers'] = (df['qualifiers']
                                 .apply(lambda x: 
                                        [{i['type']['displayName'] : i['type']['value']} for i in x]))

    # convert splitted qualifiers to dict
    df['converted_qualifiers'] = (df['splitted_qualifiers']
                                  .apply(lambda x: 
                                         {list(i.keys())[0]: list(i.values())[0] for i in x}))
    return df


def fill_na_except_coordinates(df : pd.DataFrame, cols_not_to_fill : list, fill_value : [int, str]) -> pd.DataFrame:
    for col in df.columns:
        if col not in cols_not_to_fill:
            df[col] = df[col].fillna(fill_value)
    return df

def check_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    This function checks for missing values in the first 34 columns (initial columns) of a DataFrame and returns columns
    with missing values exceeding 10%.
    
    :param df: A pandas DataFrame containing the data you want to check for missing values
    :type df: pd.DataFrame
    :returns: a pandas Series containing the columns with missing
    """
    missing_values = df[df.columns[:34]].isnull().sum().sort_values(ascending=False) / len(df) * 100
    return missing_values[missing_values > 10]

def clean_redundant_cols(df : pd.DataFrame) -> pd.DataFrame:
    """
    Removes redundant columns from a pandas DataFrame.
    
    :param df: A pandas DataFrame that you want to clean by removing redundant columns
    :type df: pd.DataFrame
    :returns: pandas DataFrame with the redundant columns dropped.
    """
    df = df.drop(columns=[redundant_cols])
    return df

def merge_statistics(events : pd.DataFrame, agg_passes : pd.DataFrame, agg_goals : pd.DataFrame, agg_shots : pd.DataFrame, key=merge_key, **kwargs):
    """
    Merges initial event data with aggregated passes, goals, and shots data
    based on a specified key.
    params:
        events : pd.DataFrame, initial event data
        agg_passes : pd.DataFrame, aggregated passes data
        agg_goals : pd.DataFrame, aggregated goals data
        agg_shots : pd.DataFrame, aggregated shots data
        key : list
        kwargs : dict
    returns: a merged pandas DataFrame
    """
    df = events.merge(agg_passes, on=key, **kwargs)
    df = df.merge(agg_goals, on=key, **kwargs)
    df = df.merge(agg_shots, on=key, **kwargs)
    return df

def consolidate_statistics(stats):
    """
    Consolidates statistics by pivoting the DataFrame and creating new columns
    for each statistic.
    :param stats: A pandas DataFrame containing statistics data
    :type stats: pd.DataFrame
    :returns: pandas DataFrame with consolidated statistics
    """
    stats['idx'] = stats.groupby('jsonName').cumcount()+1

    stats = stats.pivot_table(index=['jsonName',], columns='idx', 
                        values=[*stats.drop(columns=['jsonName', 'idx', 'leagueSeason', 'is_draw', 'is_loss', 'is_win']).columns], 
                        aggfunc='sum')

    stats = stats.sort_index(axis=1, level=0)
    stats.columns = [f'{x}_{y}' for x,y in stats.columns]
    stats = stats.reset_index()

    stats['winning_team'] = np.where(stats['outcome_1'] == 2, 1, np.where(stats['outcome_2'] == 2, 2, 0))
    return stats

def df_to_data(df):
    x = torch.tensor(df[['x', 'y']].values, dtype=torch.float)
    edge_index = torch.tensor(df[['endX', 'endY']].values, dtype=torch.long).t().contiguous()
    edge_attrs = torch.tensor(df[['zone', 'playerId']].values, dtype=torch.long)
    y = torch.tensor(df['outcome'].values, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attrs)

######### visualisation methods ##########

def plot_events_distribution(events: pd.DataFrame, **kwargs) -> None:
    event_types = pd.DataFrame(events.value_counts('eventType', normalize=True) * 100)
    event_types.reset_index(inplace=True)
    event_types.columns = ['eventType', 'percentage']
    event_types = pd.concat([event_types, pd.DataFrame({'eventType': 'Shot', 'percentage': events['isShot'].sum() / events.shape[0] * 100}, index=[0])], axis=0)
    event_types = pd.concat([event_types, pd.DataFrame({'eventType': 'Goal', 'percentage': events['isGoal'].sum() / events.shape[0] * 100}, index=[0])], axis=0)
    event_types = event_types[event_types['eventType'] != 'Pass'].sort_values('percentage', ascending=False)
    event_types = event_types[event_types['percentage'] > 0.3]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=event_types, x='eventType', y='percentage')
    plt.title('Event types distribution - without passes. Threshold: 0.3%')
    plt.ylabel('Percentage')
    plt.xlabel('Event type')
    plt.xticks(rotation=45)
    plt.show()


def draw_opta_pitch() -> tuple:
    pitch = Pitch(pitch_type='opta', pitch_color='pink', linestyle='-', goal_linestyle='-', linewidth=1, line_color='black',)
    fig, ax = pitch.draw(figsize=(12, 6))
    return fig, ax

def plot_outcomes(goals: pd.DataFrame, **kwargs) -> None:
    goals.groupby(['leagueSeason', 'is_draw']).size().unstack().plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Frequency of draws by season')
    plt.xlabel('Season')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(0, len(goals['leagueSeason'].unique())), labels=goals['leagueSeason'].unique(), rotation=30)
    plt.show()

def plot_pass_by_minute(events: pd.DataFrame, n_passes = 5000, **kwargs) -> None:
    fig, ax = draw_opta_pitch()
    (events[events['eventType'] == 'Pass']
        .sample(n_passes)
        .groupby(['teamId', 'minute', 'second'])
        .agg({'x' : 'mean', 'y' : 'mean'})
        .reset_index()
        .plot(x='x', y='y', ax=ax, kind='scatter', c='minute', cmap='coolwarm', **kwargs))
    plt.show()
    
def plot_pass_map(events: pd.DataFrame, **kwargs) -> None:
    fig, ax = draw_opta_pitch()
    passes_successful = events[(events['eventType'] == "Pass") & (events['isSuccessful'] == 1)]
    passes_unsuccessful = events[(events['eventType'] == "Pass") & (events['isSuccessful'] == 0)]
    ax.annotate('Attacking direction', xy=(70, 5), xytext=(30, 5), arrowprops=dict(arrowstyle='->', lw=2, color='white'), fontsize=14, color='white', ha='center', va='center')
    ax.plot(passes_successful['x'], passes_successful['y'], 'o', color='blue', markersize=0.5, **kwargs)
    ax.plot(passes_unsuccessful['x'], passes_unsuccessful['y'], '+', color='red', markersize=0.5, alpha=0.5, markeredgewidth=0.5, **kwargs)
    plt.title('Start coordinates of successful and unsuccessful passes', fontsize=14, color='black')
    plt.show()

def plot_success_rate(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    events['isSuccessful'].value_counts(normalize=True).plot(kind='bar', ax=ax[0], color=['blue', 'red'])
    ax[0].set_title('Unsuccessful vs Successful events distribution') 
    ax[0].set_xticklabels(['Successful', 'Unsuccessful'], rotation=0)
    events.groupby(['leagueSeason', 'isSuccessful']).size().unstack().plot(kind='bar', ax=ax[1], color=['red', 'blue'])
    ax[1].set_title('Unsuccessful vs Successful events distribution by the season in EPL')
    ax[1].set_xticklabels(['2021/2022', '2022/2023', '2023/2024'], rotation=0)
    plt.show()
