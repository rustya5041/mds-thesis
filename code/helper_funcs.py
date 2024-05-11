from libraries import *
from params import *

def pretify_json_data(directory: str) -> 'csv':
    """
    Reads JSON files in batches, concatenates them into a DataFrame,
    and writes the data to CSV files with a specific naming convention.
    
    :param directory: The `directory` parameter in the `pretify_json_data` function is a string that
    represents the path to a directory containing JSON files that you want to process and convert into a
    CSV format
    :type directory: str
    :return: The function `pretify_json_data` returns the string 'Compilation completed'.
    """
    filenames = []
    for subfloder in glob(directory):
        filenames.extend(glob(f'{subfloder}/*'))
    # reading files by batches of 200, write to csv, and then read again. 
    # add leagueSeason to the name for easy filtering
    for i in range(0, len(filenames), 200):
        print(f'Processing files {i} to {i+200}')
        df = pd.DataFrame()
        for filename in filenames[i:i+200]:
            with open(filename) as f:
                df = pd.concat([df, read_footy_json(f)])
        df.to_csv(f'epl_{i}.csv')
    return 'Compilation completed'

def load_prettified_data() -> pd.DataFrame:
    """
    Reads data from multiple CSV files with footy data into a pandas DataFrame,
    converting the 'qualifiers' column values to Python literals.
    :return: A pandas DataFrame containing data loaded from multiple CSV files named 'epl_0.csv',
    'epl_1.csv', 'epl_2.csv', 'epl_3.csv', 'epl_4.csv', and 'epl_5.csv'. The 'qualifiers' column in each
    CSV file is converted using the 'literal_eval' function.
    """
    df = pd.DataFrame()
    for i in range(6):
        df = pd.concat([df, pd.read_csv(f'epl_{i}.csv', converters={'qualifiers': literal_eval})])
        break
    return df

def extract_qualifiers(row: "pd.Series"):
    """
    This function extracts data from a pandas Series object and creates a new Series with display names
    as index and corresponding values.
    
    :param row: A pandas Series containing data in a specific format. The function `extract_data` is
    designed to extract and process information from this Series
    :type row: "pd.Series"
    :return: A pandas Series object is being returned, with the values extracted from the input row
    based on the specified logic. The values are extracted from the 'value' key if present, otherwise,
    it defaults to 1 if the 'value' key is not present but the 'type' key is present. The index of the
    Series is based on the 'displayName' key from the input row.
    """
    display_names = []
    values = []
    for item in row:
        display_names.append(item['type']['displayName'])
        if 'value' in item:
            values.append(item['value'])
        else:
            values.append(1 if 'value' in item['type'] else item['type']['value'])
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
    :return: The function `split_qualifiers` takes a pandas DataFrame `df` as input, splits the
    'qualifiers' column into a list of dictionaries, and then converts this list into a single
    dictionary. The function then adds two new columns 'splitted_qualifiers' and 'converted_qualifiers'
    to the DataFrame `df` and returns the modified DataFrame.
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


def clean_redundant_cols(df : pd.DataFrame) -> pd.DataFrame:
    """
    Removes redundant columns from a pandas DataFrame.
    
    :param df: A pandas DataFrame that you want to clean by removing redundant columns
    :type df: pd.DataFrame
    :return: The function `clean_redundant_cols` is returning a pandas DataFrame with the redundant
    columns dropped.
    """
    df = df.drop(columns=[redundant_cols])
    return df

def merge_statistics(events : pd.DataFrame, agg_passes : pd.DataFrame, agg_goals : pd.DataFrame, agg_shots : pd.DataFrame, key=merge_key, **kwargs):
    """
    Merges initial event data with aggregated passes, goals, and shots data
    based on a specified key.
    :param events: The `events` parameter is expected to be a pandas DataFrame containing data related
    to events in a sports match, such as passes, goals, shots, etc
    :type events: pd.DataFrame
    :param agg_passes: The `agg_passes` parameter is a DataFrame containing aggregated statistics
    related to passes in a sports event. It likely includes information such as total passes made, pass
    completion rates, key passes, etc. This DataFrame is being merged with the `events` DataFrame based
    on a specified key column or columns
    :type agg_passes: pd.DataFrame
    :param agg_goals: The `agg_goals` parameter in the `merge_statistics` function is expected to be a
    DataFrame containing aggregated statistics related to goals in a sports event. This DataFrame is
    merged with the `events` DataFrame based on a specified key column or columns. The function performs
    a series of merge operations to combine the
    :type agg_goals: pd.DataFrame
    :param agg_shots: The `agg_shots` parameter in the `merge_statistics` function is expected to be a
    DataFrame containing aggregated statistics related to shots in a sports event. This DataFrame is
    merged with the `events` DataFrame based on a specified key column or columns. The function performs
    a series of merges with the `
    :type agg_shots: pd.DataFrame
    :param key: The `key` parameter in the `merge_statistics` function is used as the column or columns
    on which to join the DataFrames `events`, `agg_passes`, `agg_goals`, and `agg_shots`. It specifies
    the column(s) that are common between the DataFrames and serves as
    :return: The function `merge_statistics` returns a DataFrame that is the result of merging the
    `events`, `agg_passes`, `agg_goals`, and `agg_shots` DataFrames on the specified key using the
    provided kwargs.
    """
    df = events.merge(agg_passes, on=key, **kwargs)
    df = df.merge(agg_goals, on=key, **kwargs)
    df = df.merge(agg_shots, on=key, **kwargs)
    return df

def consolidate_statistics(stats):
    stats['idx'] = stats.groupby('jsonName').cumcount()+1

    stats = stats.pivot_table(index=['jsonName',], columns='idx', 
                        values=[*stats.drop(columns=['jsonName', 'idx', 'leagueSeason', 'is_draw', 'is_loss', 'is_win']).columns], 
                        aggfunc='sum')

    stats = stats.sort_index(axis=1, level=0)
    stats.columns = [f'{x}_{y}' for x,y in stats.columns]
    stats = stats.reset_index()

    stats['winning_team'] = np.where(stats['outcome_1'] == 2, 1, np.where(stats['outcome_2'] == 2, 2, 0))
    return stats


######### visualisation methods ##########

def draw_opta_pitch() -> tuple:
    pitch = Pitch(pitch_type='opta', pitch_color='pink', linestyle='-', goal_linestyle='-', linewidth=1, line_color='black',)
    fig, ax = pitch.draw(figsize=(12, 6))
    return fig, ax

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
    # pass map for successful and unsuccessful passes
    fig, ax = draw_opta_pitch()
    passes_successful = events[(events['eventType'] == "Pass") & (events['isSuccessful'] == 1)]
    passes_unsuccessful = events[(events['eventType'] == "Pass") & (events['isSuccessful'] == 0)]
    ax.plot(passes_successful['x'], passes_successful['y'], 'o', color='blue', markersize=0.5, **kwargs)
    ax.plot(passes_unsuccessful['x'], passes_unsuccessful['y'], '+', color='red', markersize=0.5, alpha=0.5, markeredgewidth=0.5, **kwargs)
    plt.show()

def plot_success_rate(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # plot successful/unsuccessful passes as a bar chart in %
    events['isSuccessful'].value_counts(normalize=True).plot(kind='bar', ax=ax[0], color=['blue', 'red'])
    ax[0].set_title('Unsuccessful vs Successful events distribution') 
    ax[0].set_xticklabels(['Successful', 'Unsuccessful'], rotation=0)

    # plot successful/unsuccessful passes by leagueSeason
    events.groupby(['leagueSeason', 'isSuccessful']).size().unstack().plot(kind='bar', ax=ax[1], color=['red', 'blue'])
    ax[1].set_title('Unsuccessful vs Successful events distribution by the season in EPL')
    ax[1].set_xticklabels(['2021/2022', '2022/2023', '2023/2024'], rotation=0)
    plt.show()
