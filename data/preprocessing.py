import pandas as pd
import pathlib

from itertools import product

FRAME_FREQUENCY = 10 # [1/s] 
# use to convert consecutive timestamps into delta time [s]

# data source
inpath = pathlib.Path(r'./trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.txt')
# headers
colnames = ['t', 'id', 'x', 'y']


with open(inpath) as infile:
    df = pd.read_csv(infile, sep='\t', names=colnames)
    
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
        df.loc[rows.index, 'dt'] = rows['t'].diff() / FRAME_FREQUENCY
        # convert delta position to velocity
        df['dx'] /= df['dt']
        df['dy'] /= df['dt']

    # replace NaN elements with zero
    df = df.fillna(0)
    # note: velocities for the agents that just arrived in the scene are set to zero,
    # even thought there is no way for us to know what the position of the agent was
    # before it arrived on the scene.

    # save
    df.to_csv(inpath.parent / (inpath.stem + '.csv'))