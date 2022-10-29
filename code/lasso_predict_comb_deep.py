from lx_commonfunc1 import *
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV
import xgboost as xgb
import optuna
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
#feat_name = 'lstm'
def main(feat_name, exp_path):
    for wfid in [1]:

        dl = Dataloader_self(wfid, norm_method,  window_size=time_len, channel=channel)
        input_shape = dl.dnn_data_onesample().shape

        wtnumber = dl.wtnumber
        # Candidate hyperparams
        saving_path = 'wf' + str(wfid) + '/' + str(exp_path)
        saving_path2 = 'wf' + str(wfid) + '/' + str(exp_path2)
        saving_path3 = 'wf' + str(wfid) + '/' + str(exp_path3)
        pt_path = 'wf' + str(wfid) + '/sae/sae_md'
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
        tr = []
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
            clf = MultiTaskLassoCV().fit(tX, tY)
            pred_teY=clf.predict(teX)        
            out_renorm = dl.renorm(pred_teY)
            true_renorm = dl.renorm(teY)
            pr.append(out_renorm)
            x_min = np.min(true_renorm, axis=1)
            index = np.where(x_min > 50)[0]

            RMSE_test_mat[wt] = RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
            MAPE_test_mat[wt] = MAPE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])

            data_save(RMSE_test_mat, saving_path + '/RMSE_test_mat_' + feat_name + '_lasso_comb.csv')
            data_save(MAPE_test_mat, saving_path + '/MAPE_test_mat_' + feat_name + '_lasso_comb.csv')
            mask = np.sum(np.abs(clf.coef_), axis=0) != 0
            mask = mask.reshape(-1)
            a += np.sum(mask[:len1])
            b += np.sum(mask[len1:])
        print(a, b)    
        pr=np.concatenate(pr, axis=0)
        f = open(saving_path + '/' + feat_name + '_lasso_comb_pred.txt', 'wb')
        pickle.dump(pr, f)


if __name__ == '__main__':
    main()
