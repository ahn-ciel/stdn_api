import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

class Trainer():
    def __init__(self, model, args, train_loader=None, val_loader=None, test_loader=None):
        self.args = args
        self.model = model
        self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        # self.optimizer = optim.Adam(self.model.parameters(), lr = args.learning_rate)

    def train(self):
        min_val = 1e3
        patience = 5
        wait = 0
        s1 = time.time()
        times = []
        logging.info("Training started...")
        for epoch in range(1, self.args.epochs + 1):
            epoch_s = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self.eval(val = True)
            times.append(time.time()-epoch_s)
            logging.info(f"Epoch {epoch:03d} train complete in {times[-1]:.2f} secs, Train Loss: {train_loss * self.args.volume_train_max:.4f}, Val Loss: {val_loss * self.args.volume_train_max:.4f}")
            if val_loss < min_val:
                min_val = val_loss
                wait = 0
                save_path = f"{self.args.save_dir}/STDN_{self.args.name}_{epoch:03d}_{val_loss * self.args.volume_train_max:.3f}.pth"
                torch.save(self.model.state_dict(), save_path)
            else:
                wait += 1
            if (wait > patience) and (epoch > 40):
                print(f"Early Termination on epoch {epoch:03d}!")
                break
        logging.info(f"Average Training Time per Epoch: {np.mean(times):.2f} secs")
        logging.info(f"Total Training Time: {time.time()-s1:.2f} secs")
        return save_path

    def _train_epoch(self, epoch):
        self.model.train()
        loader = self.train_loader
        losses = []
        for i, (att_cnn_x, att_flow, att_x, cnn_x, flow, x, y) in enumerate(loader):
            train_xs = [torch.Tensor(input).float().to(self.args.device) for input in (att_cnn_x, att_flow, att_x, cnn_x, flow, x)]
            train_y = torch.Tensor(y).float().to(self.args.device)
            self.optimizer.zero_grad()
            pred = self.model(*train_xs)
            loss = self.loss(pred, train_y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if (i + 1) % self.args.print_every == 0:
                logging.info(f"Iter {i+1:05d}, Loss: {np.mean(losses) * self.args.volume_train_max:.4f}")
        return np.mean(losses)

    def eval(self, val = True):
        def criterion(pred, y):
            threshold = self.args.threshold
            mask = y > threshold
            mape = torch.mean(torch.abs((pred - y) / y)[mask])
            rmse = torch.sqrt(torch.mean((pred[mask] - y[mask])**2))
            return mape, rmse
        self.model.eval()
        if val:
            loader = self.val_loader
        else:
            loader = self.test_loader
        losses = []
        mapes = []
        rmses = []
        for i, (att_cnn_x, att_flow, att_x, cnn_x, flow, x, y) in enumerate(loader):
            test_xs = [torch.Tensor(input).float().to(self.args.device) for input in (att_cnn_x, att_flow, att_x, cnn_x, flow, x)]
            test_y = torch.Tensor(y).float().to(self.args.device)
            pred = self.model(*test_xs)
            loss = self.loss(pred, test_y)
            losses.append(loss.item())
            if not val:
                mape, rmse = criterion(pred, test_y)
                mapes.append(mape.item())
                rmses.append(rmse.item())
        if not val:
            return np.mean(losses), 100 * np.mean(mapes), np.mean(rmses)
        return np.mean(losses)

    def test(self):
        self.model.eval()
        loader = self.test_loader
        preds = []
        for i, (att_cnn_x, att_flow, att_x, cnn_x, flow, x, y) in enumerate(loader):
            test_xs = [torch.Tensor(input).float().to(self.args.device) for input in (att_cnn_x, att_flow, att_x, cnn_x, flow, x)]
            test_y = torch.Tensor(y).float().to(self.args.device)
            pred = self.model(*test_xs)
            preds.append(pred.detach().cpu().numpy())
        
        return np.concatenate(preds, axis = 0)