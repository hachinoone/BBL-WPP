from lx_commonfunc5 import *
import copy

exp_path = 'feature_selection'

channel = 'time'
filter_size = 4
layer_num = 1
channel_num = 128
pool_method = 'max'
norm_method = 'maxmin'
fc_dim = 128
batch_size = 64
learn_rate = 0.001
lr_decay = 0.8
hidden_size = 16
pred_len = 6
n_feat=13
n_wt=33
scale=85
time_len=60
device = 'cuda:3'
loss_func = nn.MSELoss()
# print_step = 10
# epochs = 100
print_step = 10
epochs = 150


def result_test(wt, test_emb, batch_size, md, dl, saving_path, target_use, RMSE_mat, MAPE_mat):
    train, train_10, batch_number = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use=target_use)


    ind = np.arange(batch_size).reshape(batch_size, 1).repeat(pred_len, axis=1)
    for i in range(1,pred_len):
        ind[:,i] = ind[:,0] + i
    ind = ind.reshape(-1)
    ind2 = np.arange(batch_size).reshape(batch_size, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        ind2[:,i] = ind2[:,0] + i
    ind2 = ind2.reshape(-1)

    add = (np.arange(batch_size) * scale).reshape(batch_size, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        add[:,i] = add[:,0] + i

    left = train_10.shape[0]-time_len-pred_len-batch_size*(batch_number-1)
    
    idx_l_10 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(pred_len, axis=1)
    for i in range(pred_len):
        idx_l_10[:,i] = idx_l_10[:,0] + i - time_len

    idx_l = (batch_size*(batch_number-1) * scale + np.arange(left) * scale).reshape(left, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        idx_l[:,i] = idx_l[:,0] + i

    idx_l_2 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        idx_l_2[:,i] = idx_l_2[:,0] + i


    for p in range(batch_number):

        if (p < batch_number - 1):
            idx = add + batch_size * p * scale
            idx_2 = ind2 + batch_size * p

            x_train2 = torch.tensor(train_10[idx_2.reshape(-1), wt], dtype = torch.float, device=device).view(batch_size, time_len, n_feat)
            idx = ind + batch_size * p + time_len
            y_train = train_10[idx.reshape(-1), wt, 0].reshape(batch_size, pred_len)
        else:

            x_train2 = torch.tensor(train_10[idx_l_2.reshape(-1), wt], dtype = torch.float, device=device).view(left, time_len, n_feat)
            y_train = train_10[idx_l_10.reshape(-1), wt, 0].reshape(left, pred_len)

        if p == 0:
            out = md(x_train2, device).cpu().detach().data.numpy()
            emb = md.emb(x_train2, device).cpu().detach().data.numpy()
            true = y_train
        else:
            out = np.concatenate([out, md(x_train2, device).cpu().detach().data.numpy()], axis=0)
            emb = np.concatenate([emb, md.emb(x_train2, device).cpu().detach().data.numpy()], axis=0)
            true = np.concatenate([true, y_train], axis=0)
    out_renorm = dl.renorm(out)
    true_renorm = dl.renorm(true)
    x_min = np.min(true_renorm, axis=1)
    index = np.where(x_min > 50)[0]

    RMSE = RMSE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
    MAPE = MAPE_compute(pred=out_renorm[index, :], true=true_renorm[index, :])
    pickle.dump(emb, open(saving_path + '/emb_lstm_' + target_use + str(wt), 'wb'))

    print("RMSE is " + str(RMSE))
    print("MAPE is " + str(MAPE))
    RMSE_mat[wt, :] = RMSE
    MAPE_mat[wt, :] = MAPE

    data_save(RMSE_mat, saving_path + '/RMSE_'+ target_use +'single_mat.csv')
    data_save(MAPE_mat, saving_path + '/MAPE_'+ target_use +'single_mat.csv')
    return RMSE_mat, MAPE_mat

def result_val(wt, val_emb, batch_size, md, dl, target_use):
    train, train_10, batch_number = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use=target_use)

    ind = np.arange(batch_size).reshape(batch_size, 1).repeat(pred_len, axis=1)
    for i in range(1,pred_len):
        ind[:,i] = ind[:,0] + i
    ind = ind.reshape(-1)
    ind2 = np.arange(batch_size).reshape(batch_size, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        ind2[:,i] = ind2[:,0] + i
    ind2 = ind2.reshape(-1)

    add = (np.arange(batch_size) * scale).reshape(batch_size, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        add[:,i] = add[:,0] + i

    left = train_10.shape[0]-time_len-pred_len-batch_size*(batch_number-1)
    
    idx_l_10 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(pred_len, axis=1)
    for i in range(pred_len):
        idx_l_10[:,i] = idx_l_10[:,0] + i - time_len

    idx_l = (batch_size*(batch_number-1) * scale + np.arange(left) * scale).reshape(left, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        idx_l[:,i] = idx_l[:,0] + i

    idx_l_2 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        idx_l_2[:,i] = idx_l_2[:,0] + i


    for p in range(batch_number):
        if (p < batch_number - 1):
            idx = add + batch_size * p * scale
            idx_2 = ind2 + batch_size * p
            x_train2 = torch.tensor(train_10[idx_2.reshape(-1), wt], dtype = torch.float, device=device).view(batch_size, time_len, n_feat)
            idx = ind + batch_size * p + time_len
            y_train = train_10[idx.reshape(-1), wt, 0].reshape(batch_size, pred_len)
        else:
            x_train2 = torch.tensor(train_10[idx_l_2.reshape(-1), wt], dtype = torch.float, device=device).view(left, time_len, n_feat)
            y_train = train_10[idx_l_10.reshape(-1), wt, 0].reshape(left, pred_len)

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


def model_train(wt, train_emb, val_emb, test_emb, batch_size, model, dl, learn_rate, lr_decay, print_step, saving_path, epochs):
    train_error = np.zeros([epochs])
    val_error = np.zeros([epochs])
    test_error = np.zeros([epochs])
    best_score = 1e7
    best_model = None
    train, train_10, batch_number = dl.dnn_data_generator(tgwtid=wt, batch_size=batch_size, target_use='train')
    #print(pt.shape, train_10.shape)

    model = model.to(device)
    ind = np.arange(batch_size).reshape(batch_size, 1).repeat(pred_len, axis=1)
    for i in range(1,pred_len):
        ind[:,i] = ind[:,0] + i
    ind = ind.reshape(-1)
    ind2 = np.arange(batch_size).reshape(batch_size, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        ind2[:,i] = ind2[:,0] + i
    ind2 = ind2.reshape(-1)
    add = (np.arange(batch_size) * scale).reshape(batch_size, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        add[:,i] = add[:,0] + i

    left = train_10.shape[0]-time_len-pred_len-batch_size*(batch_number-1)
    
    idx_l_10 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(pred_len, axis=1)
    for i in range(pred_len):
        idx_l_10[:,i] = idx_l_10[:,0] + i - time_len

    idx_l = (batch_size*(batch_number-1) * scale + np.arange(left) * scale).reshape(left, 1).repeat(time_len*scale, axis=1)
    for i in range(1,time_len*scale):
        idx_l[:,i] = idx_l[:,0] + i

    idx_l_2 = (batch_size*(batch_number-1) + np.arange(left)).reshape(left, 1).repeat(time_len, axis=1)
    for i in range(1,time_len):
        idx_l_2[:,i] = idx_l_2[:,0] + i

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** (epoch//5))
    for epoch in range(epochs):
        avg_loss = 0.
        counter = 0
        index = random.sample(range(batch_number), batch_number)
        model.train()
        for p in range(batch_number):
            if (index[p] < batch_number - 1):
                idx_2 = ind2 + batch_size * index[p]
                x_train2 = torch.tensor(train_10[idx_2.reshape(-1), wt], dtype = torch.float, device=device).view(batch_size, time_len, n_feat)
                idx = ind + batch_size * index[p] + time_len
                y_train = torch.tensor(train_10[idx.reshape(-1), wt, 0], dtype=torch.float, device=device).view(batch_size, pred_len)
            else:
                x_train2 = torch.tensor(train_10[idx_l_2.reshape(-1), wt], dtype = torch.float, device=device).view(left, time_len, n_feat)
                y_train = torch.tensor(train_10[idx_l_10.reshape(-1), wt, 0], dtype=torch.float, device=device).view(left, pred_len)

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
        score = result_val(wt, val_emb, batch_size, model, dl, 'vali')

        val_error[epoch] = score
        if (score < best_score):
            best_score = score
            best_model = copy.deepcopy(model)
            if ~torch.isnan(loss):
                torch.save(best_model.state_dict(), saving_path + "/single_dnn_wt" + str(wt) + ".pkl")
            else:
                print('error!!!!')
                sys.exit(0)

        score2 = result_val(wt, test_emb, batch_size, model, dl, 'test')
        test_error[epoch] = score2
        #print(avg_loss)
        data_save(train_error, saving_path+'/single_loss_record_wt'+str(wt)+'.csv')
        data_save(val_error, saving_path+'/single_val_record_wt'+str(wt)+'.csv')
        data_save(test_error, saving_path+'/single_test_record_wt'+str(wt)+'.csv')

        if epoch % print_step == 0:
            print("Epoch {}......... Average Loss for Epoch: {}".format(epoch, avg_loss / counter))
            print("Epoch {}......... Average Val Score for Epoch: {}".format(epoch, score))
            print("Epoch {}......... Average Test Score for Epoch: {}".format(epoch, score2))
    return best_model

def main():
    for wfid in [1]:
        dl = Dataloader_self(wfid, norm_method,  window_size=time_len, channel=channel)
        input_shape = dl.dnn_data_onesample().shape
        wtnumber = dl.wtnumber
        # Candidate hyperparams
        saving_path = 'wf' + str(wfid) + '/' + str(exp_path)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        RMSE_train_mat = np.zeros([wtnumber, pred_len])
        MAPE_train_mat = np.zeros([wtnumber, pred_len])
        RMSE_vali_mat = np.zeros([wtnumber, pred_len])
        MAPE_vali_mat = np.zeros([wtnumber, pred_len])
        RMSE_test_mat = np.zeros([wtnumber, pred_len])
        MAPE_test_mat = np.zeros([wtnumber, pred_len])
        train_time = np.zeros([wtnumber])
        for wt in range(wtnumber):
            print("The " + str(wt) + "-th WT")
            train_emb, val_emb, test_emb = None, None, None
            model = LSTM(time_len, n_feat, hidden_size).to(device)
            #print('wt', wt)
            model.train()
            model = model.to(device)
            start_time = time.time()
            print("Model training")
            model = model_train(wt, train_emb, val_emb, test_emb, batch_size, model, dl, learn_rate, lr_decay, print_step, saving_path, epochs)
            model.eval()
            RMSE_train_mat, MAPE_train_mat = result_test(wt, train_emb, batch_size, model, dl, saving_path,
                                                       target_use='train', RMSE_mat=RMSE_train_mat,
                                                       MAPE_mat=MAPE_train_mat)
            RMSE_vali_mat, MAPE_vali_mat = result_test(wt, val_emb, batch_size, model, dl, saving_path,
                                                       target_use='vali', RMSE_mat=RMSE_vali_mat,
                                                       MAPE_mat=MAPE_vali_mat)
            RMSE_test_mat, MAPE_test_mat = result_test(wt, test_emb, batch_size, model, dl, saving_path,
                                                       target_use='test', RMSE_mat=RMSE_test_mat,
                                                       MAPE_mat=MAPE_test_mat)

if __name__ == '__main__':
    main()