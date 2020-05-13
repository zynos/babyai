import os
import pickle
import babyai
import gym
import matplotlib.pyplot as plt
import torch
from myScripts.ReplayBuffer import ProcessData
from myScripts.supervisedNet import Net
from torch.nn.utils import clip_grad_value_
from myScripts.Rudder import Rudder
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import datetime


class Training:

    def __init__(self):

        self.grad_norms = []
        self.rudder = Rudder()
        self.device = "cuda"
        self.use_widi_lstm = True
        self.action_only = False
        self.rudder.device = self.device
        self.rudder.net = Net(image_dim=128, instr_dim=128, ac_embed_dim=128, action_space=7, device=self.device,
                              use_widi=self.use_widi_lstm, action_only=self.action_only).to(self.device)
        self.rudder.use_transformer = True
        self.rudder.mu = 1
        self.rudder.quality_threshold = 0.8
        self.rudder.clip_value = 0.5
        self.lr = lr = 1e-6
        self.weight_dec = 1e-6
        self.rudder.optimizer = torch.optim.Adam(self.rudder.net.parameters(), lr=self.lr, weight_decay=self.weight_dec)
        self.rudder.use_transformer = False
        self.epochs = 10

    # def train_and_set_metrics(self, episode):
    #     # loss, returnn, quality = self.train_one_episode(episode)
    #     loss, returns, quality, predictions =self.rudder.feed_network(episode)
    #
    #     ### APEX
    #     # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #     #     scaled_loss.backward()
    #     ###
    #     loss.backward()
    #     grad_norm = sum(
    #         p.grad.data.norm(2) ** 2 for p in self.rudder.net.parameters() if p.grad is not None) ** 0.5
    #     clip_grad_value_(self.rudder.net.parameters(), self.clip_value)
    #     self.grad_norms.append(grad_norm.item())
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #
    #     loss = loss.detach().item()
    #     returnn = returns.detach().item()
    #     # print("Loss",loss)
    #     episode.loss = loss
    #     episode.returnn = returnn
    #     # print("loss", loss)
    #     return quality, predictions

    def plot(self, returns, train_losses, test_losses):
        plt.title(
            "aO"+str(self.action_only)+ " TR"+str(self.rudder.use_transformer)+" widi " + str(self.use_widi_lstm) + " aux loss 0.01 return mean {:.2f} lr {} w_dec {} epochs {}".format(
                np.mean(returns), self.lr, self.weight_dec,
                self.epochs))
        returns = np.array(returns)
        plt.ylabel("loss")
        plt.xlabel("episodes")
        plt.plot(train_losses, label="train loss")
        plt.plot(test_losses, label="test loss")
        plt.legend(loc="upper left")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(19.2, 9.83)
        plt.savefig("trainResult", dpi=100)
        plt.show()

    def random_train_test_split(self, episodes):
        random.shuffle(episodes)
        split_index = int(len(episodes) * 0.8)
        train = episodes[:split_index]
        test = episodes[split_index:]
        return train, test

    def train(self):
        episodes = read_pkl_files(False)
        # episodes = episodes[:2]
        get_return_mean(episodes)
        train, test = self.random_train_test_split(episodes)
        train_losses = []
        test_losses = []
        returns = []

        for i in range(self.epochs):
            print(i,datetime.datetime.now().time())
            tmp_loss = []
            for ep in train:
                _, _, ep, _ = self.rudder.train_and_set_metrics(ep)
                tmp_loss.append(ep.loss)
                returns.append(ep.returnn)
            epoch_loss = np.mean(tmp_loss)
            print("train loss", epoch_loss)
            train_losses.append(epoch_loss)
            tmp_loss = []
            for ep in test:
                self.rudder.inference_and_set_metrics(ep)
                tmp_loss.append(ep.loss)
                returns.append(ep.returnn)
            epoch_loss = np.mean(tmp_loss)
            print("test loss", epoch_loss)
            test_losses.append(epoch_loss)
            fname="MyModel"+str(i)+".pt"
            torch.save(self.rudder.net.state_dict(),fname )
            self.rudder.net.load_state_dict(torch.load(fname))


        self.plot(returns, train_losses, test_losses)
        torch.save(self.rudder.net.state_dict(), "MyModel.pt")

    def plot_reward_redistribution(self, orig_rews, redistributed_rews, actions, ax, i,label,plot_orig):
        action_dict = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up", 4: "drop", 5: "toggle",
                       6: "done"}
        actions = [action_dict[a.item()] for a in actions]
        rews = orig_rews.cpu().numpy()
        redistributed_rews = redistributed_rews.cpu().squeeze().numpy()
        if plot_orig:
            ax.plot(rews, label="original rewards")
        ax.plot(redistributed_rews, label="redistributed rewards "+str(label))
        ax.set_xticks(list(range(len(actions))))
        ax.set_xticklabels(actions, rotation=90)
        ax.legend(loc="upper right")
        ax.stem([i], [redistributed_rews[i]], linefmt="r--", markerfmt="r")
        ax.get_xticklabels()[i].set_color("red")
        return actions[i]
        # plt.show()

    def get_predictions_from_different_models(self,short_episode):
        path = "models/"
        files = os.listdir(path)
        ret =[]
        for f in files:
            print(f)
            if "widi" in f:
                self.rudder.net = Net(image_dim=128, instr_dim=128, ac_embed_dim=128, action_space=7, device=self.device,
                                      use_widi=True, action_only=self.action_only).to(self.device)
            else:
                self.rudder.net = Net(image_dim=128, instr_dim=128, ac_embed_dim=128, action_space=7,
                                      device=self.device,
                                      use_widi=False, action_only=self.action_only).to(self.device)
            # if "trans" in f:
            #     self.rudder.net = Net(image_dim=128, instr_dim=128, ac_embed_dim=128, action_space=7,
            #                           device=self.device,
            #                           use_widi=False, action_only=self.action_only).to(self.device)
            self.rudder.net.load_state_dict(torch.load(path+f))
            loss, returns, quality, predictions = self.rudder.feed_network(short_episode)
            ret.append((predictions.squeeze(),f[:-3]))
        return ret


    def evaluate(self):
        env = gym.make("BabyAI-PutNextLocal-v0")
        # self.rudder.net.load_state_dict(torch.load("MyModel.pt"))
        episodes = read_pkl_files(True)
        random.shuffle(episodes)
        short_episode = None
        fist = False
        for e in episodes:
            e: ProcessData
            if 20 < len(e.dones) <60:
                short_episode = e
                break
                # if fist:
                #     break
                # fist = True
        model_predictions = self.get_predictions_from_different_models(short_episode)
        # loss, returns, quality, predictions = self.rudder.feed_network(short_episode)
        command = {"put": 1, "the": 2, "grey": 3, "key": 4, "next": 5, "to": 6, "red": 7, "box": 8, "yellow": 9,
                   "blue": 10, "green": 11, "purple": 12, "ball": 13}
        inv_map = {v: k for k, v in command.items()}
        lis = [inv_map[i.item()] for i in short_episode.instructions[0].cpu()]
        command = " ".join(lis)
        print(command)
        for i, image in enumerate(short_episode.images):
            # plt.figure(figsize=(19.20,9.83))

            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(2, 1, figsize=(19.20, 9.83))

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically

            renderer = env.render("human")
            r = env.get_obs_render(image.cpu().numpy(), 128)
            # predictions = predictions.squeeze()
            plot_orig=True
            for el in model_predictions:

                action = self.plot_reward_redistribution(short_episode.rewards,el[0], short_episode.actions,
                                                         axarr[0], i,el[1],plot_orig)
                plot_orig=False

            fname = "myPics/testImag" + str(i)
            r.toImage().save(fname, "PNG")
            arr = mpimg.imread(fname)
            axarr[1].imshow(arr)
            axarr[1].title.set_text("next action: " + str(action))
            axarr[0].title.set_text(command)
            plt.tight_layout()
            plt.savefig("myPics/coolPic" + str(i), dpi=100)
            plt.close()
            # plt.gcf().close()
            # plt.show()
            # r=r.scaledToHeight(256)


def read_pkl_file2s():
    all_episodes = []
    path = "../scripts/replays/"
    files = os.listdir(path)
    for numb in range(len(files)):
        with open(path + "rep" + str(numb) + ".pkl", "rb") as f:
            episodes = pickle.load(f)
            all_episodes.extend(episodes)
            break
    print("episodes", len(all_episodes))
    return all_episodes


def read_pkl_files(evaluate):
    all_episodes = []
    if evaluate:
        path = "../scripts/replays2/"
    else:
        path = "../scripts/replays/"
    files = os.listdir(path)
    for file in files:
        with open(path + file, "rb") as f:
            episodes = pickle.load(f)
            all_episodes.extend(episodes)
    print("episodes", len(all_episodes))
    return all_episodes


def get_return_mean(episodes):
    rets = []
    for episode in episodes:
        rets.append(torch.sum(episode.rewards).item())
    print("mean return", np.mean(rets))


# env = gym.make("BabyAI-PutNextLocal-v0")
training = Training()
training.train()
