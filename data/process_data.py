import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load .csv files and merge data for messages and corresponding categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,
                        'left',
                        left_on = 'id',
                        right_on = 'id')
    return df

def clean_data(df):

    # split categorie field and add the result to the dataframe
    df = pd.concat([df[['id', 'message','original','genre']], 
                    df.categories.str.split(pat = ';', expand = True)], axis = 1)
    
    # extract first row to get categories names
    row = list(df.iloc[0,4:])
    trim_2last_character = lambda x: x[:-2]
    category_colnames = [trim_2last_character(x) for x in row]

    # convert string data in categories to binary
    for column in list(df.columns)[4:]:
    # set each value to be the last character of the string
        df[column] = df[column].astype(str).str.slice(start = -1)

        # convert column from string to numeric
        df[column] = pd.to_numeric(df[column])
        df[column] = np.where(df[column]==0, 0, 1)
    
    # remove duplicates
    df.columns = ['id', 'message', 'original', 'genre'] + category_colnames
    df.drop_duplicates(subset = ['message'], inplace = True)
    
    return df
    
def save_data(df, database_filename):
    """
    Write df dataframe to messages_treated table in the database_filename.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_treated', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()