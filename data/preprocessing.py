import pandas as pd
import pathlib

from itertools import product

DELTA_TIMESTAMP = 0.1 # 10 milllis in between frames 

path = pathlib.Path(r'.\trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.txt') # data source

colnames = ['t', 'id', 'x', 'y'] # headers

with open(path) as f:
    df = pd.read_csv(f, delimiter='\t', names=colnames)
    
    timestamps = df['t'].unique() 
    ids = df['id'].unique()

    idxs = list(product(timestamps, ids))
    
    df = df.set_index(['t', 'id']).reindex(idxs).reset_index()

    for id in ids:
        # select rows for a specific id
        rows = df.loc[df['id'] == id]
        # add delta position between consecutive frames
        df.loc[rows.index, 'dx'] = rows['x'].diff()
        df.loc[rows.index, 'dy'] = rows['y'].diff()
        # add delta time between consecutive frames
        df.loc[rows.index, 'dt'] = rows['t'].diff() * DELTA_TIMESTAMP
        # convert delta position to velocity
        df['dx'] /= df['dt']
        df['dy'] /= df['dt']

    # replace NaN elements with zero
    df = df.fillna(0)
    # note: velocities for the agents that just arrived in the scene are set to zero,
    # even though there is no way for us to know what the position of the agent was
    # before it arrived on the scene.

    # save
    df.to_csv(path.parent / (path.stem + '.csv'), index=False)