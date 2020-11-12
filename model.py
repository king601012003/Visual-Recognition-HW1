import os
import numpy as np
from ResNet import ResNet18, ResNet50
from dataloader import CVLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.models as torch_model
import matplotlib.pyplot as plt
import pandas as pd
from efficientnet_pytorch import EfficientNet
# from ranger import Ranger

def train_it(batch_data, batch_label, net, loss_function, optimizer):
    batch_data = batch_data.float().cuda()
    batch_label = batch_label.long()
    
    batch_datas, batch_labels, batch_labelss, lam = mixup_data(batch_data, batch_label)
    
    net.train()
    
    loss = 0
    prediction = net(batch_data).cpu()
    
    loss = mixup_criterion(loss_function, prediction, batch_labels, batch_labelss, lam)
    
    # loss = loss_function(prediction, batch_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def eval_it(data, label, net):
    data = data.float().cuda()
    label = label.long()
    
    net.eval()
    
    prediction = np.argmax(F.softmax(net(data).cpu(), dim=1).data.numpy(), axis=1)
    acc = np.mean(np.equal(prediction.data,label.data.numpy()))
    
    return acc

def eval_one_weight(data, label, net):
    data = data.float().cuda()
    label = label.long()
    
    net.eval()
        
    prediction = np.argmax(F.softmax(net(data).cpu(), dim=1).data.numpy(), axis=1)
    acc = np.mean(np.equal(prediction.data,label.data.numpy()))
    
    return acc, np.asarray(prediction.data), label.data.numpy()

def submit_one_weight(data, net):
    data = data.float().cuda()
    
    net.eval()
        
    prediction = np.argmax(F.softmax(net(data).cpu(), dim=1).data.numpy(), axis=1)
    
    return np.asarray(prediction.data)

def get_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    plt.matshow(df_confusion, cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')



def get_torch_model(model_name, fix_weight=False):
    if model_name == "restnet18":        
        net = torch_model.resnet18(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, 196)
        
    elif model_name == "restnet50": 
        net = torch_model.resnet50(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, 196)
    
    elif model_name == "efficientnet": 
        net = EfficientNet.from_pretrained('efficientnet-b5')
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net._fc = nn.Linear(net._fc.in_features, 196)
    elif model_name == "wide_resnet50_2": 
        net = torch_model.wide_resnet50_2(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, 196)
        
    return net

def adjust_lr(optimizer, epoch):
    
    if epoch <= 20:
        lr = 1e-4
    # elif (epoch > 2) and (epoch <= 4):
    #     lr = 5e-5
    # elif (epoch > 12) and (epoch <= 30):
    #     lr = 1e-5
    elif (epoch > 20) and (epoch <= 50):
        lr = 5e-5
    else:
        lr = 1e-5
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
      
    data_train = CVLoader("./","train")   
    data_test = CVLoader("./","test") 
    data_submit = CVLoader("./","submit") 

#####################################################  Hyperparameters #####################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    network = 8
    batch_size = 25
    stop_epoch = 300
    model_state = "submit"  # train, eval, submit
    model_weight = "epoch_85.pkl"
    tensorboard_path = "./tensorboard/loss_nolr"
    ckpt_path = "ckpt"
#####################################################  Hyperparameters #####################################################    
    
    if network == 1:
        net = ResNet18()
    elif network == 2:
        net = ResNet50()
    elif network == 3:
        net = get_torch_model("restnet18")
    elif network == 4:
        net = get_torch_model("restnet50")
    elif network == 5:
        net = get_torch_model("restnet18", fix_weight=True)
    elif network == 6:
        net = get_torch_model("restnet50", fix_weight=True)
    elif network == 7:
        net = get_torch_model("wide_resnet50_2")
    elif network == 8:
        net = get_torch_model("efficientnet")
        
 
    net.cuda()
    
    if network == 1 or network == 2:
        # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    elif network == 3 or network == 4:
        # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    elif network == 5 or network == 6:
        # optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    elif network == 7 or network == 8:
        # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        # optimizer = Ranger(net.parameters())
    
    loss_function = nn.CrossEntropyLoss()
    highest_test_acc = 0
    it = 0  
    
    if model_state == "train":
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        tb_log = SummaryWriter(log_dir=tensorboard_path)  
        
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4)
        
        for epoch in range(1, stop_epoch):
            # adjust_lr(optimizer, epoch)
            for cur_it, (batch_data, batch_label) in enumerate(train_loader):
                
                loss = train_it(batch_data, batch_label, net, loss_function, optimizer)
                
                tb_log.add_scalar('loss', loss, it)
                it += 1
                print("================Epoch: ", epoch, "||  Batch: ", cur_it, "================")
                print("Loss: %.5f" % loss.data.numpy())
                
                train_acc = 0
                train_acc = eval_it(batch_data, batch_label, net)
                tb_log.add_scalar('train_acc', train_acc, it)
                print("train_acc: %.4f" % train_acc)
                print("highest_test_acc: %.4f" % highest_test_acc)
                
            if epoch % 1 == 0:
                
                print("===============Start eval testing data=================")
                
                test_acc = 0

                for cur_it, (batch_data, batch_label) in enumerate(test_loader):
                    test_acc += eval_it(batch_data, batch_label, net)

                    
                test_acc = test_acc / (cur_it + 1)
                
                if test_acc > highest_test_acc :
                    if not os.path.isdir("./" + ckpt_path):
                        os.makedirs("./" + ckpt_path)
                    save_name = "./" + ckpt_path + "/epoch_" + str(epoch) + ".pkl"
                    
                    if isinstance(net, nn.DataParallel):
                        torch.save(net.module.state_dict(), save_name)
                    else:
                        torch.save(net.state_dict(), save_name)
                
                highest_test_acc = max(highest_test_acc, test_acc)
                
                print("test_acc: %.4f " % test_acc)
                print("highest_test_acc: %.4f" % highest_test_acc)
                    
                tb_log.add_scalar('test_acc', test_acc, epoch)
                
                

    elif model_state == "eval":
        
        print("start loading weight...")
        net.load_state_dict(torch.load("./" + ckpt_path + "/" + model_weight))
        print("finish loading weight")
        
        # train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4)
        # train_acc = 0
        test_acc = 0
        test_accs = 0
        # for cur_it, (batch_data, batch_label) in enumerate(train_loader):
        #     train_acc += eval_one_weight(batch_data, batch_label, net)
        # train_acc = train_acc / (cur_it + 1)
            
        for cur_it, (batch_data, batch_label) in enumerate(test_loader):
            test_acc, part_predic, part_label = eval_one_weight(batch_data, batch_label, net)
            test_accs += test_acc
            if cur_it == 0:
                total_predic = part_predic*1
                total_label = part_label*1
            else:
                total_predic = np.concatenate((total_predic,part_predic), axis=0)
                total_label = np.concatenate((total_label,part_label), axis=0)
            
        test_accs = test_accs / (cur_it + 1)

        print("result of " + model_weight)
        # print("train_acc: %.4f" % train_acc)
        print("test_acc: %.4f " % test_accs)
        
        y_actu = pd.Series(total_label.tolist(), name='Actual')
        y_pred = pd.Series(total_predic.tolist(), name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        df_confusion = df_confusion.div(df_confusion.sum(axis=1),axis=0)
        get_confusion_matrix(df_confusion)
    
    # summary(net, input_size=(3,224,224))
    
    elif model_state == "submit":
        
        print("start loading weight...")
        net.load_state_dict(torch.load("./" + ckpt_path + "/" + model_weight))
        print("finish loading weight")
        submit_loader = DataLoader(data_submit, batch_size=batch_size, shuffle=True, num_workers=4)
        
        for cur_it, (batch_data,img_name) in enumerate(submit_loader):
            print("Batch:", cur_it)
            submit_prediction = submit_one_weight(batch_data, net)
            
            if cur_it == 0:
                temp_prediction = submit_prediction.copy()
                temp_img_name = img_name.numpy().copy()
                
            else:
                temp_prediction = np.concatenate((temp_prediction,submit_prediction))
                temp_img_name = np.concatenate((temp_img_name, img_name.numpy().copy()))

        output_array = np.vstack((temp_img_name,temp_prediction)).T
        
        df = pd.DataFrame(output_array, columns = ["id","label"])
        
        df.to_csv("./submit_result.csv", index=False)
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
