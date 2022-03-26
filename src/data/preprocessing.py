from pathlib import Path


# from itertools import product

# DELTA_TIMESTAMP = 0.1 # 10 milllis in between frames 

path = Path(r'.\trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.txt') # data source

# colnames = ['t', 'id', 'x', 'y'] # headers

# df = pd.read_csv(path, delimiter='\t', names=colnames)

# ids = df['id'].unique()
# for id in ids:
#     # select rows for a specific id
#     rows = df.loc[df['id'] == id]
#     # add delta position between consecutive frames
#     df.loc[rows.index, 'dx'] = rows['x'].diff()
#     df.loc[rows.index, 'dy'] = rows['y'].diff()
#     # add delta time between consecutive frames
#     df.loc[rows.index, 'dt'] = rows['t'].diff() * 0.1
#     # convert delta position to velocity
#     df['dx'] /= df['dt']
#     df['dy'] /= df['dt']

# with open(path) as f:
#     df = pd.read_csv(f, delimiter='\t', names=colnames)
    
#     timestamps = df['t'].unique() 
#     ids = df['id'].unique()

#     idxs = list(product(timestamps, ids))
    
#     df = df.set_index(['t', 'id']).reindex(idxs).reset_index()

#     for id in ids:
#         # select rows for a specific id
#         rows = df.loc[df['id'] == id]
#         # add delta position between consecutive frames
#         df.loc[rows.index, 'dx'] = rows['x'].diff()
#         df.loc[rows.index, 'dy'] = rows['y'].diff()
#         # add delta time between consecutive frames
#         df.loc[rows.index, 'dt'] = rows['t'].diff() * DELTA_TIMESTAMP
#         # convert delta position to velocity
#         df['dx'] /= df['dt']
#         df['dy'] /= df['dt']

#     # replace NaN elements with zero
#     df = df.fillna(0)
#     # note: velocities for the agents that just arrived in the scene are set to zero,
#     # even though there is no way for us to know what the position of the agent was
#     # before it arrived on the scene.

#     # save
# df.to_csv(path.parent / (path.stem + '.csv'), index=False)






# class IDK(object):

#     def __init__(self, model) -> None:
#         if not isinstance(model, nn.Module):
#             raise NotImplementedError
#         self._model = model


#         # add device / cuda AND model = model.to(device)



#     def __call__(self, x):
#         # simple forward pass of the model with input x: return y = self._model(x)
#         pass

    


#     def train(train_loader, net, optimizer, criterion):
#         """
#         Trains network for one epoch in batches.

#         Args:
#             train_loader: Data loader for training set.
#             net: Neural network model.
#             optimizer: Optimizer (e.g. SGD).
#             criterion: Loss function (e.g. cross-entropy loss).
#         """
    
#         avg_loss = 0
#         correct = 0
#         total = 0

        # # iterate through batches
        # for i, data in enumerate(train_loader):
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = data

        #     # zero the parameter gradients
        #     optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # keep track of loss and accuracy
    #         avg_loss += loss
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     return avg_loss/len(train_loader), 100 * correct / total


    # def train(self, train_loader, optimizer, criterion, device):
    #     pass

    # def test():
    #     pass

    # def evaluate():
    #     pass

    # def run(self):
    #     pass