import pandas as pd
import glob
import os

from src.utils.constants import GTZAN_DIR


def get_df_gtzan(path: str = GTZAN_DIR):
    df_parse = pd.DataFrame(columns=['genre', 'number', 'path'])

    for p in glob.glob(os.path.join(str(path), '*', '*.wav')):
        fname = p.split(os.sep)[-1]

        genre, number, _ = fname.split('.')

        df_parse.loc[len(df_parse)] = [genre, number, p]

    return df_parse.reset_index(drop=True)

