from lx_commonfunc1 import *

from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV

import xgboost as xgb

exp_path = 'feature_selection'

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

def main():
    for wfid in [1]:

        dl = Dataloader_self(wfid, norm_method,  window_size=time_len, channel=channel)
        input_shape = dl.dnn_data_onesample().shape

        wtnumber = dl.wtnumber

        saving_path = 'wf' + str(wfid) + '/' + str(exp_path)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        RMSE_train_mat = np.zeros([wtnumber, pred_len])
        MAPE_train_mat = np.zeros([wtnumber, pred_len])
        RMSE_vali_mat = np.zeros([wtnumber, pred_len])
        MAPE_vali_mat = np.zeros([wtnumber, pred_len])
        RMSE_test_mat = np.zeros([wtnumber, pred_len])
        MAPE_test_mat = np.zeros([wtnumber, pred_len])

        pr = []
        train_time = np.zeros([wtnumber])
        for wt in range(wtnumber):
            print("The " + str(wt) + "-th WT")
            print("Model training")
            train_x_all, train_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='train')
            trX = np.concatenate(train_x_all, axis=0)
            trX = trX[:,:,wt].reshape(trX.shape[0], -1)
            trY = np.concatenate(train_y_all, axis=0)
            vali_x_all, vali_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='vali')
            vX = np.concatenate(vali_x_all, axis=0)
            vX = vX[:,:,wt].reshape(vX.shape[0], -1)
            vY = np.concatenate(vali_y_all, axis=0)
            tX = np.concatenate((trX, vX), axis=0)
            tY = np.concatenate((trY, vY), axis=0)
            test_x_all, test_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='test')
            teX = np.concatenate(test_x_all, axis=0)
            teX = teX[:,:,wt].reshape(teX.shape[0], -1)
            teY = np.concatenate(test_y_all, axis=0)
            clf = MultiTaskLassoCV().fit(tX, tY)
            mask = np.sum(np.abs(clf.coef_), axis=0) != 0
            mask = mask.reshape(-1)
            savepath = saving_path + '/mask_lasso_single_wt_' + str(wt)
            with open(savepath, 'wb') as f:
                pickle.dump(mask, f)
if __name__ == '__main__':
    main()
