from lx_commonfunc1 import *
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import optuna
def rmseloss(output, target):
    return np.sqrt(np.mean((output-target)**2))


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
feat_thres = 0.9
time_len=60
device = 'cuda:3'
loss_func = nn.MSELoss()
# print_step = 10
# epochs = 100
print_step = 10
epochs = 100

max_depth_set = [5, 30, 50]
min_samples_leaf_set = [50, 200, 400]
n_estimators_set = [10, 20, 30]

params = {'max_depth': 5, 'min_samples_leaf': 50, 'n_estimators': 10}

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
        a, b = 0, 0

        train_time = np.zeros([wtnumber])
        pr = []
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

            best_params = None
            best_cost = 1e7

            for max_depth in max_depth_set:
                for min_samples_leaf in min_samples_leaf_set:
                    for n_estimators in n_estimators_set:
                        params['max_depth'] = max_depth
                        params['min_samples_leaf'] = min_samples_leaf
                        params['n_estimators'] = n_estimators
                        clf = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                        clf.fit(trX, trY)
                        pred_vY=clf.predict(vX)
                        out_renorm = dl.renorm(pred_vY)
                        true_renorm = dl.renorm(vY)
                        cost = np.mean(RMSE_compute(pred=out_renorm, true=true_renorm))
                        if (cost < best_cost):
                            best_cost = cost
                            best_params = params
            clf = RandomForestRegressor(max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'], n_estimators=best_params['n_estimators']).fit(tX,tY)
            ipt = clf.feature_importances_
            feat = ipt.argsort()
            wt_sum = 0.
            for i in range(ipt.size - 1, -1, -1):
                wt_sum = wt_sum + ipt[feat[i]]
                if wt_sum >= feat_thres:
                    break
            ind = feat[i:]
            mask = np.zeros(feat.shape[0]).astype(int)
            mask[ind] = 1
            mask = (mask != 0)
            savepath = saving_path + '/mask_rf_single_wt_' + str(wt)
            with open(savepath, 'wb') as f:
                pickle.dump(mask, f)


if __name__ == '__main__':
    main()
