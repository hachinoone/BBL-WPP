from lx_commonfunc1 import *
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV
import xgboost as xgb
import optuna
from sklearn.ensemble import RandomForestRegressor
def rmseloss(output, target):
    return np.sqrt(np.mean((output-target)**2))

#exp_path = 'prediction'
exp_path2 = 'train_alllstmattn'
exp_path3 = 'feature_selection'

channel = 'time'
filter_size = 4
layer_num = 1
channel_num = 128
pool_method = 'max'
norm_method = 'maxmin'
fc_dim = 128
batch_size = 256
learn_rate = 0.0025
lr_decay = 0.85
hidden_size = 32
pred_len = 6
n_feat=13
n_wt=33

time_len=60
device = 'cuda:3'
loss_func = nn.MSELoss()
# print_step = 10
# epochs = 100
print_step = 10
epochs = 100
#feat_name = 'cnn'



def result_test(trX, trY, batch_size, md, saving_path, target_use, RMSE_mat, MAPE_mat, dl, wt):
    batch_number = int(np.ceil(trX.shape[0] / batch_size)) 
    left = trX.shape[0] - (batch_number-1) * batch_size
    for p in range(batch_number):
        if (p < batch_number - 1):
            x_train2 = torch.tensor(trX[p*batch_size:(p+1)*batch_size], dtype = torch.float, device=device).view(batch_size, -1)
            y_train = trY[p*batch_size:(p+1)*batch_size]
        else:
            x_train2 = torch.tensor(trX[p*batch_size:(p+1)*batch_size], dtype = torch.float, device=device).view(left, -1)
            y_train = trY[p*batch_size:(p+1)*batch_size]
        if p == 0:
            out = md(x_train2, device).cpu().detach().data.numpy()
            true = y_train
        else:
            out = np.concatenate([out, md(x_train2, device).cpu().detach().data.numpy()], axis=0)
            true = np.concatenate([true, y_train], axis=0)
    out_renorm = dl.renorm(out)
    true_renorm = dl.renorm(true)
    x_min = np.min(true_renorm, axis=1)
    index = np.where(x_min > 50)[0]

    RMSE = RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
    MAPE = MAPE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])

    print("RMSE is " + str(RMSE))
    print("MAPE is " + str(MAPE))
    RMSE_mat[wt, :] = RMSE
    MAPE_mat[wt, :] = MAPE

    data_save(RMSE_mat, saving_path + '/RMSE_' + target_use + '_mat_' + feat_name + '_dnn_comb.csv')
    data_save(MAPE_mat, saving_path + '/MAPE_' + target_use + '_mat_' + feat_name + '_dnn_comb.csv')
    return RMSE_mat, MAPE_mat

def result_val(trX, trY, batch_size, md, dl):
    batch_number = int(np.ceil(trX.shape[0] / batch_size)) 
    left = trX.shape[0] - (batch_number-1) * batch_size
    for p in range(batch_number):
        if (p < batch_number - 1):
            x_train2 = torch.tensor(trX[p*batch_size:(p+1)*batch_size], dtype = torch.float, device=device).view(batch_size, -1)
            y_train = trY[p*batch_size:(p+1)*batch_size]
        else:
            x_train2 = torch.tensor(trX[p*batch_size:(p+1)*batch_size], dtype = torch.float, device=device).view(left, -1)
            y_train = trY[p*batch_size:(p+1)*batch_size]
        if p == 0:
            out = md(x_train2, device).cpu().detach().data.numpy()
            true = y_train
        else:
            out = np.concatenate([out, md(x_train2, device).cpu().detach().data.numpy()], axis=0)
            true = np.concatenate([true, y_train], axis=0)
    out_renorm = dl.renorm(out)
    true_renorm = dl.renorm(true)
    x_min = np.min(true_renorm, axis=1)
    index = np.where(x_min > 50)[0]

    RMSE = RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])

    return np.mean(RMSE)


def model_train(trX, trY,vX, vY, teX, teY, batch_size, model, dl, learn_rate, lr_decay, print_step, saving_path, epochs):
    train_error = np.zeros([epochs])
    val_error = np.zeros([epochs])
    test_error = np.zeros([epochs])
    best_score = 1e7
    best_model = None
    batch_number = int(np.ceil(trX.shape[0] / batch_size)) 
    left = trX.shape[0] - (batch_number-1) * batch_size
    #print(pt.shape, train_10.shape)

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** (epoch//5))
    for epoch in range(epochs):
        avg_loss = 0.
        counter = 0
        index = random.sample(range(batch_number), batch_number)
        model.train()
        for p in range(batch_number):
            if (index[p] < batch_number - 1):
                x_train2 = torch.tensor(trX[index[p]*batch_size:(index[p]+1)*batch_size], dtype = torch.float, device=device).view(batch_size, -1)
                y_train = torch.tensor(trY[index[p]*batch_size:(index[p]+1)*batch_size], dtype=torch.float, device=device).view(batch_size, pred_len)
            else:
                x_train2 = torch.tensor(trX[index[p]*batch_size:(index[p]+1)*batch_size], dtype = torch.float, device=device).view(left, -1)
                y_train = torch.tensor(trY[index[p]*batch_size:(index[p]+1)*batch_size], dtype=torch.float, device=device).view(left, -1)
            out = model(x_train2, device)
            model.zero_grad()
            loss = loss_func(out, y_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            counter += 1
            avg_loss += loss.item()


        lr_scheduler.step()
        train_error[epoch] = avg_loss
        model.eval()
        score = result_val(vX, vY, batch_size, model, dl)

        val_error[epoch] = score
        if (score < best_score):
            best_score = score
            best_model = model
            if ~torch.isnan(loss):
                pass
            else:
                print('error!!!!')
                sys.exit(0)

    return best_model


def main(feat_name, exp_path):
    for wfid in [1]:
        dl = Dataloader_self(wfid, norm_method,  window_size=time_len, channel=channel)
        input_shape = dl.dnn_data_onesample().shape

        wtnumber = dl.wtnumber
        saving_path = 'wf' + str(wfid) + '/' + str(exp_path)
        saving_path2 = 'wf' + str(wfid) + '/' + str(exp_path2)
        saving_path3 = 'wf' + str(wfid) + '/' + str(exp_path3)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        RMSE_train_mat = np.zeros([wtnumber, pred_len])
        MAPE_train_mat = np.zeros([wtnumber, pred_len])
        RMSE_vali_mat = np.zeros([wtnumber, pred_len])
        MAPE_vali_mat = np.zeros([wtnumber, pred_len])
        RMSE_test_mat = np.zeros([wtnumber, pred_len])
        MAPE_test_mat = np.zeros([wtnumber, pred_len])
        pr = []
        feat = np.zeros(2)
        train_time = np.zeros([wtnumber])
        for wt in range(wtnumber):
            print("The " + str(wt) + "-th WT")
            print("Model training")
            f = open(saving_path3 + '/emb_'+ feat_name + '_train' + str(wt), 'rb')
            train_emb = pickle.load(f)
            f.close()
            f = open(saving_path3 + '/emb_'+ feat_name + '_vali' + str(wt), 'rb')
            val_emb = pickle.load(f)
            f.close()
            f = open(saving_path3 + '/emb_'+ feat_name + '_test' + str(wt), 'rb')
            test_emb = pickle.load(f)

            f = open(saving_path2 + '/emb_wt_train' + str(wt), 'rb')
            trX_p = pickle.load(f)
            f.close()
            f = open(saving_path2 + '/emb_wt_vali' + str(wt), 'rb')
            vX_p = pickle.load(f)
            f.close()
            f = open(saving_path2 + '/emb_wt_test' + str(wt), 'rb')
            teX_p = pickle.load(f)
            f.close()

            train_x_all, train_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='train')
            trX = np.concatenate((train_emb, trX_p), axis=1)
            len1, len2 = train_emb.shape[1], trX_p.shape[1]
            trY = np.concatenate(train_y_all, axis=0)

            f.close()
            vali_x_all, vali_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='vali')
            vX = np.concatenate((val_emb, vX_p), axis=1)
            vY = np.concatenate(vali_y_all, axis=0)
            tX = np.concatenate((trX, vX), axis=0)
            tY = np.concatenate((trY, vY), axis=0)
            test_x_all, test_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='test')
            teX = np.concatenate((test_emb, teX_p), axis=1)
            teY = np.concatenate(test_y_all, axis=0)
            model = DNN(trX.shape[1], hidden_size, pred_len).to(device)
            #print('wt', wt)
            model.train()
            model = model.to(device)
        
            start_time = time.time()
            print("Model training")
            model = model_train(trX, trY,vX, vY, teX, teY, batch_size, model, dl, learn_rate, lr_decay, print_step, saving_path, epochs)
            model.eval()
            RMSE_test_mat, MAPE_test_mat = result_test(teX, teY, batch_size, model, saving_path,
                                                       target_use='test', RMSE_mat=RMSE_test_mat,
                                                       MAPE_mat=MAPE_test_mat, dl=dl, wt=wt)

if __name__ == '__main__':
    main()

