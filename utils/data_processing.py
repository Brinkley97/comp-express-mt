import os

import pandas as pd

class DataProcessing:

    def load_csv(file):
        df = pd.read_csv(file)
        return df

    def load_data(file: str):
        """Load any data given the file path. Supported ['.csv']"""
        _, extension = os.path.splitext(file)

        if extension == '.csv':
            return DataProcessing.load_csv(file)
        else:
            print(f"File extension {extension} is NOT supported yet. Choose from ['.csv'].")


    def drop_data_from_df(df: pd.DataFrame, drop_cols: list = None, dropna: bool = False) -> pd.DataFrame:
        """
        Optionally drop specified columns and/or rows with NaN values.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to clean.
            drop_cols (list, optional): List of column names to drop.
            dropna (bool, optional): Whether to drop rows with any NaN values.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        
        if dropna:
            df.dropna(inplace=True)
        
        return df
    
    def rename_df_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        """Specify the columns to rename in a DF."""
        df.rename(columns=cols, inplace=True)
        return df
    
    def split_df_mappings(df: pd.DataFrame, idx: list) -> pd.DataFrame:
        """Split the One-to-Many and Many-to-One mappings"""
        return df.iloc[: , idx]
    
    def fill_na_with_value(df: pd.DataFrame, col_name: str) -> list:
        """Fill NaN values in a column with the last valid value."""
        filled_values = []
        last_valid = None

        for _, row in df.iterrows():
            value = row[col_name]
            if pd.isna(value):
                value = last_valid
            else:
                last_valid = value
            filled_values.append(value)

        return filled_values

    def df_to_pivot_table(df: pd.DataFrame, col_to_update: str, index_columns: list) -> pd.DataFrame:
        """Fill missing values in a column and return a pivot table using specified index columns."""
        df[col_to_update] = DataProcessing.fill_na_with_value(df, col_to_update)
        updated_df = df[index_columns]
        pivot = pd.pivot_table(updated_df, index=index_columns, aggfunc='first')
        return pivot

    def save_data(df: pd.DataFrame, file: str, index: bool = True):
        """Load any data given the file path. Supported ['.csv']"""
        _, extension = os.path.splitext(file)

        if extension == '.csv':
            df.to_csv(file, index=index)
        else:
            print(f"File extension {extension} is NOT supported yet. Choose from ['.csv'].")
