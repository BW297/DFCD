import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error, f1_score
from utils import get_doa


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.device = config['device']

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        raise NotImplementedError("method must be implemented in subclasses.")

    def get_mastery_level(self):
        raise NotImplementedError("method must be implemented in subclasses.")

    def monotonicity(self):
        raise NotImplementedError("Method must be implemented in subclasses.")

    def train_step(self):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        all_epoch = self.config['epoch']
        total_auc, total_ap, total_acc, total_f1, total_rmse, total_doa  = [], [], [], [], [], []
        for epoch_i in range(self.config['epoch']):
            epoch_losses = []
            for batch_data in tqdm(self.config['train_dataloader'], "Epoch %s" % epoch_i):
                student_id, exercise_id, knowledge_point, y = [data.to(self.device) for data in batch_data]

                pred = self.forward(student_id, exercise_id, knowledge_point)
                bce_loss = nn.BCELoss()(pred, y)
                total_loss = bce_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.monotonicity()

                epoch_losses.append(total_loss.item())

            # print("[Epoch %d] average loss: %.6f" % (epoch_i, np.mean(epoch_losses)))

            auc, ap, acc, rmse, f1, doa = self.test_step()
            metrics_dict = {
                'auc': auc,
                'ap': ap,
                'acc': acc,
                'rmse': rmse,
                'f1': f1,
                'doa': doa
            }
            total_auc.append(auc)
            total_ap.append(ap)
            total_acc.append(acc)
            total_rmse.append(rmse)
            total_f1.append(f1)
            total_doa.append(doa)
            print(f'[{epoch_i:03d}/{all_epoch}] | Loss: {np.mean(epoch_losses):.4f} | AUC: {auc:.4f} | '
                  f'ACC: {acc:.4f} | RMSE: {rmse:.4f} | F1: {f1:.4f} | DOA@10: {doa:.4f}')
            
        print(f'Best AUC: {max(total_auc)}, Best AP: {max(total_ap)}, Best ACC:{max(total_acc)}, Best RMSE:{min(total_rmse)}, Best F1-Score:{max(total_f1)}, Best Doa:{max(total_doa)}')

    @torch.no_grad()
    def test_step(self):
        self.eval()
        new_preds, preds, new_ys, ys = [], [], [], []
        for batch_data in tqdm(self.config['test_dataloader'], "Testing"):
            student_id, exercise_id, knowledge_point, y = [data.to(self.device) for data in batch_data]
            pred = self.forward(student_id, exercise_id, knowledge_point, mode='eval').cpu().numpy().tolist()
            new_pred, new_y = [], []
            if self.config['split'] == 'Stu':
                for idx, student in enumerate(student_id):
                    if student.detach().cpu().numpy() in self.config['new_idx']:
                        new_pred.append(pred[idx])
                        new_y.append(y.cpu().numpy().tolist()[idx])
            elif self.config['split'] == 'Exer' or self.config['split'] == 'Know':
                for idx, exercise in enumerate(exercise_id):
                    if exercise.detach().cpu().numpy() in self.config['new_idx']:
                        new_pred.append(pred[idx])
                        new_y.append(y.cpu().numpy().tolist()[idx])
            preds.extend(pred)
            ys.extend(y.cpu().numpy().tolist())
            new_preds.extend(new_pred)
            new_ys.extend(new_y)

        if self.config['split'] == 'Stu' or self.config['split'] == 'Exer' or self.config['split'] == 'Know':
            return roc_auc_score(new_ys, new_preds), average_precision_score(new_ys, new_preds), accuracy_score(new_ys, np.array(
            new_preds) >= 0.5), np.sqrt(mean_squared_error(new_ys, new_preds)), f1_score(new_ys, np.array(new_preds) >= 0.5), get_doa(
            self.config,
            self.get_mastery_level())
        else:
            return  roc_auc_score(ys, preds), average_precision_score(ys, preds), accuracy_score(ys, np.array(
            preds) >= 0.5), np.sqrt(mean_squared_error(ys, preds)), f1_score(ys, np.array(preds) >= 0.5), get_doa(
            self.config,
            self.get_mastery_level())