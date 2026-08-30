import pandas as pd
import os

def load_and_preprocess(file_path):
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    

    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_life']
    
    df = df.merge(max_cycle, on=['unit_nr'], how='left')
    df['RUL'] = df['max_life'] - df['time_cycles']
    df.drop('max_life', axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    path = 'data/raw/train_FD001.txt'
    if os.path.exists(path):
        processed_df = load_and_preprocess(path)
        if not os.path.exists('data/processed'): os.makedirs('data/processed')
        processed_df.to_csv('data/processed/train_cleaned.csv', index=False)
        print("Step 1 Complete: Data processed and saved!")
    else:
        print("Error: 'train_FD001.txt' file nahi mili. Check path: data/raw/")