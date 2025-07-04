import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from SOH.utils.util import AverageMeter,get_logger,eval_metrix
import os
import wandb
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self,input_dim=6,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=6,output_dim=32,layers_num=3,hidden_dim=60,droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        batch_size, small_batch, input_dim = x.shape
        x = x.reshape(-1, input_dim)  # (32 * 126, 6)
        x = self.encoder(x)
        x = x.reshape(batch_size, small_batch, -1)  # (32, 126, 32)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out
    

class CNN(nn.Module):
    '''
    input shape: (batch_size, 126, 40) 
    output shape: (batch_size, 32) 
    '''
    def __init__(self):
        super(CNN,self).__init__()
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(40, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU()
        ) # 输出: (N,24,126)

        # 残差块
        self.layer1 = ResBlock(24, 32, stride=2)  # 输出: (N,32,63)
        self.layer2 = ResBlock(32, 48, stride=2)  # 输出: (N,48,32)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):   
        '''
        :param x: shape:(batch_size, 126, 40)
        :return: features shape:(batch_size, 32)
        '''
        x = x.transpose(1, 2)  # (batch_size, 40, 126)
        # 初始卷积
        out = self.conv1(x)    # (N,24,126)
        # 残差块
        out = self.layer1(out) # (N,32,63)
        out = self.layer2(out) # (N,48,32)
        # 全局池化和特征提取
        out = self.global_pool(out)        # (N,48,1)
        features = self.feature_extractor(out.squeeze(-1)) # (N,32)
        return features


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr



class PINN(nn.Module):
    def __init__(self,args):
        super(PINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(input_dim=33,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               droupout=0.2).to(device)
        self.feature_extractor = CNN().to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.optimizer3 = torch.optim.Adam(self.feature_extractor.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.wandb = self.args.wandb

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])

        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt).sum(dim=1)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1) 
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        # dimxt = (batch_size, 126, 20)
        xt.requires_grad = True
        x = xt[:,:, 1:]  #(batch_size, 126, 19)
        t = xt[:,:, 0].unsqueeze(2)  #(batch_size, 126, 1) 每个循环的充电时间

        charge_time = t.sum(dim=1) # (batch_size, 1) 所有循环的充电时间

        u = self.solution_u(torch.cat((x,t),dim=2)) # (batch_size, 126, 20) -> (batch_size, 126, 1)
        u = u.sum(dim=1) # (batch_size, 1)
        u_t = grad(u.sum(), t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0] # (batch_size, 126, 1)
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0] # (batch_size, 126, 19)
        
        xt_u_xt = torch.cat([xt, u_t, u_x], dim=2) # shape: (batch_size, 126, 40)
        xt_u_xt_feat = self.feature_extractor(xt_u_xt)  # shape: (batch_size, 32)

        F = self.dynamical_F(torch.cat([u,xt_u_xt_feat],dim=1)) # inputdim = 1+32=33

        f = u/charge_time - F # 这个cycle的soh_loss/总充电时间 - F 要约等于0
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)
            # 用寿命衰减总和作比较

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            self.optimizer3.step()
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            ################################## info ##############################################
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)

            if self.args.wandb:
                wandb.log({
                    "epoch": e,
                    "learning_rate": current_lr,
                    "train_total_loss": loss1 + self.alpha*loss2 + self.beta*loss3,
                    "train_data_loss": loss1,
                    "train_PDE_loss": loss2,
                    "train_physics_loss": loss3,
                })
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
                if self.args.wandb:
                    wandb.log({
                        "epoch": e,
                        "valid_mse": valid_mse
                    })

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                if self.args.wandb:
                    wandb.log({
                        "epoch": e,
                        "test_mse": MSE,
                        "test_mae": MAE,
                        "test_mape": MAPE,
                        "test_rmse": RMSE
                    })
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict(),
                                   'feature_extractor':self.feature_extractor.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))




if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()


    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)




