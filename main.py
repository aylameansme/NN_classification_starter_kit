#import
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

#customs
from utils import *
from model import *

# settings
set_random(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[✓] GPU Check : {device}")
#hyperparams ( if you need, use yaml file or argparse )
n_epoch = 100
bs = 32

lr_ = 0.02
b1_ = 0.5
b2_ = 0.99


#dataloader
datlst = [i for i in os.listdir("./data/") if i.endswith(".npy")]
for j in datlst :
    globals()[f'{j[:-4]}'] = np.load(f"./data/{j}")

#train/ valid split
tr_x_, vl_x_, tr_y_, vl_y_ = train_test_split(tr_x, tr_y, test_size=0.1, random_state=42, stratify=tr_y)

#To tensor
## for train
tr_x_ = torch.from_numpy(tr_x_).float()
tr_y_ = torch.from_numpy(tr_y_).long()
vl_x_ = torch.from_numpy(vl_x_).float()
vl_y_ = torch.from_numpy(vl_y_).long()
## for test
ts_x_ = torch.from_numpy(ts_x).float()
ts_y_ = torch.from_numpy(ts_y).long()

tr_tensor = TensorDataset(tr_x_, tr_y_)
vl_tensor = TensorDataset(vl_x_, vl_y_)

ts_tensor = TensorDataset(ts_x_)

tr_loader = DataLoader(tr_tensor, batch_size=bs, shuffle=True)
vl_loader = DataLoader(vl_tensor, batch_size=bs, shuffle=True)

ts_loader = DataLoader(ts_tensor, batch_size=bs, shuffle=False)


# load model
model = DNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr_, betas=(b1_, b2_))
criterion = nn.CrossEntropyLoss().cuda()

#train
for ep in range(n_epoch):
    model.train()

    for bidx, batch in enumerate(zip(tr_loader)):
        #split x and y
        bx, by = batch[0]

        # Reset gradient
        optimizer.zero_grad()

        log = model(bx.cuda())
        loss = criterion(log, by.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #validation ( if you don't need, skip this phase.)
    with torch.no_grad():
        model.eval()

        vl_acc, vl_loss = evaluation(model, vl_loader, criterion)
        if vl_loss.type()=='torch.FloatTensor':
            vl_loss = vl_loss.tolist()

        if ep % 10 ==0:
            print(f"epoch : {ep+1}/{n_epoch}  ")
            print(f"[☁︎] validation acc : {round(vl_acc*100,2)} %")
            print(f"[☁︎] validation loss : {round(vl_loss,4)}")

#torch.save(model.state_dict(), './model_log/model.pkl')

print("-"*30)
print("             Done!")
print("-"*30)
#test
"""
labels of test dataset are not given. please export the predicted labels as predicted.csv and submit it. 
"""
#with torch.no_grad():
    #model.eval()

    #correct = 0
    #total = 0
    #for test, true_label in ts_loader :
    #    log = model(test.cuda())
    #    _, predicted = torch.max(F.softmax(log, -1).data, -1)
    #
    #    total += true_label.size(0)
    #    correct += (predicted == true_label.cuda()).sum().item()

    #print(f"Accuracy of the model on the test data : {round(100*correct/total,2)} ")