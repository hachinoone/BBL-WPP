from lx_commonfunc1 import *
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV
import xgboost as xgb
import optuna
from sklearn.ensemble import RandomForestRegressor
def rmseloss(output, target):
    return np.sqrt(np.mean((output-target)**2))

class mutiDimensionXGB():
    def __init__(self, params):
        self.params = params

    def fit(self, x, y):
        steps = y.shape[1]
        self.steps = steps
        self.model = []
        for i in range(steps):
            md = xgb.XGBRegressor(**self.params)
            md.fit(x, y[:, i])
            self.model.append(md)

    def predict(self, x):
        for i in range(self.steps):
            if i == 0:
                yp = self.model[i].predict(x).reshape((-1, 1))
            else:
                yp = np.hstack((yp, self.model[i].predict(x).reshape((-1, 1))))
        return yp

    def feature_importances(self):
        z = np.zeros(self.model[0].feature_importances_.shape[0])
        for i in range(self.steps):
            z = z + self.model[i].feature_importances_
        return z / 6

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
#feat_name = 'lstm'

max_depth_set = [5, 8, 10]
n_estimators_set = [10, 30, 50]
gamma_set = [0.7, 1.0, 1.5]
params = {'max_depth': 5, 'learning_rate':0.25, 'n_estimators':15, 'objective':'reg:squarederror',
           'nthread': -1, 'gamma':0.3, 'min_child_weight':1, 'max_delta_step':0, 'subsample':0.7,
           'colsample_bytree':0.7, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1}

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
        a, b = 0, 0

        pr = []
        feat = np.zeros(2)
        train_time = np.zeros([wtnumber])
        for wt in range(wtnumber):
            print("The " + str(wt) + "-th WT")
            print("Model training")
            maskpath = saving_path3 + '/mask_' + feat_name + '_single_wt_' + str(wt)
            print(maskpath)
            with open(maskpath, 'rb') as f:
            mask = pickle.load(f)
            train_x_all, train_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='train')
            trX = np.concatenate(train_x_all, axis=0)
            trX = trX[:,:,wt].reshape(trX.shape[0], -1)
            trX = trX[:,mask.flatten()]
            trY = np.concatenate(train_y_all, axis=0)

            f = open(saving_path2 + '/emb_wt_train' + str(wt), 'rb')
            trX_p = pickle.load(f)
            len1, len2 = trX.shape[1], trX_p.shape[1]
            trX = np.concatenate((trX, trX_p), axis=1)
            f.close()
            vali_x_all, vali_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='vali')
            vX = np.concatenate(vali_x_all, axis=0)
            vX = vX[:,:,wt].reshape(vX.shape[0], -1)
            vX = vX[:,mask.flatten()]
            vY = np.concatenate(vali_y_all, axis=0)
            f = open(saving_path2 + '/emb_wt_vali' + str(wt), 'rb')
            vX_p = pickle.load(f)
            vX = np.concatenate((vX, vX_p), axis=1)
            f.close()
            tX = np.concatenate((trX, vX), axis=0)
            tY = np.concatenate((trY, vY), axis=0)
            test_x_all, test_y_all = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='test')
            teX = np.concatenate(test_x_all, axis=0)
            teX = teX[:,:,wt].reshape(teX.shape[0], -1)
            teX = teX[:,mask.flatten()]
            teY = np.concatenate(test_y_all, axis=0)
            f = open(saving_path2 + '/emb_wt_test' + str(wt), 'rb')
            teX_p = pickle.load(f)
            teX = np.concatenate((teX, teX_p), axis=1)
            f.close()
            best_params = None
            best_cost = 1e7
            for max_depth in max_depth_set:
                for n_estimators in n_estimators_set:
                    for gamma in gamma_set:
                        params['max_depth'] = max_depth
                        params['n_estimators'] = n_estimators
                        params['gamma'] = gamma
                        clf = mutiDimensionXGB(params)
                        clf.fit(trX, trY)                
                        pred_vY=clf.predict(vX)
                        out_renorm = dl.renorm(pred_vY)
                        true_renorm = dl.renorm(vY)
                        x_min = np.min(true_renorm, axis=1)
                        index = np.where(x_min > 50)[0]
                        cost = np.mean(RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :]))
                        if (cost < best_cost):
                            best_cost = cost
                            best_params = params
            clf = mutiDimensionXGB(best_params)
            clf.fit(tX, tY)
            pred_teY=clf.predict(teX)        
            out_renorm = dl.renorm(pred_teY)
            true_renorm = dl.renorm(teY)
            pr.append(out_renorm)
            x_min = np.min(true_renorm, axis=1)
            index = np.where(x_min > 50)[0]

            RMSE_test_mat[wt] = RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
            MAPE_test_mat[wt] = MAPE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
            data_save(RMSE_test_mat, saving_path + '/RMSE_test_mat_' + feat_name + '_xgb_comb.csv')
            data_save(MAPE_test_mat, saving_path + '/MAPE_test_mat_' + feat_name + '_xgb_comb.csv')
        
            ipt = clf.feature_importances()
            feat[0] = feat[0] + np.sum(ipt[:len1])
            feat[1] = feat[1] + np.sum(ipt[len1:])
        print(feat)
        np.savetxt(saving_path + '/' + feat_name + '_xgb_comb_feat.txt', feat)
        pr=np.concatenate(pr, axis=0)
        f = open(saving_path + '/' + feat_name + '_xgb_comb_pred.txt', 'wb')
        pickle.dump(pr, f)

if __name__ == '__main__':
    main()

