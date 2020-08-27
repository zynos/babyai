import os
import pickle
import re
import sys
import time
from collections import Counter
import babyai
import gym
import matplotlib.pyplot as plt
import torch
import babyai.rl
from babyai.utils.demos import transform_demos
from myScripts.ReplayBuffer import ProcessData
from myScripts.supervisedNet import Net
from torch.nn.utils import clip_grad_value_
from myScripts.Rudder import Rudder
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import datetime
from pathlib import Path

# _ = torch.manual_seed(123)
# random.seed(1234)


class Training:

    def __init__(self, use_transformer=False):

        self.out_image_height = 10.8
        self.total_pics = 0
        self.grad_norms = []
        self.rudder = Rudder()
        self.rudder.reward_scale = 1
        self.rudder.minus_to_one_scale = True
        self.device = "cuda"
        self.image_dim = 128
        self.instr_dim = 128
        self.use_widi_lstm = False
        self.use_widi_uninit = False
        self.use_gru = False
        self.action_only = False
        self.rudder.use_transformer = use_transformer
        self.rudder.transfo_upgrade = False
        self.rudder.aux_loss_multiplier = 0.5
        assert sum([self.use_widi_uninit,self.use_widi_lstm, self.use_gru, self.rudder.use_transformer]) <= 1

        self.rudder.device = self.device
        self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128, action_space=7,
                              device=self.device,
                              use_widi=self.use_widi_lstm, action_only=self.action_only,
                              use_transformer=self.rudder.use_transformer, use_gru=self.use_gru,
                              transfo_upgrade=self.rudder.transfo_upgrade,use_unit_widi=self.use_widi_uninit).to(self.device)
        self.rudder.mu = 1
        self.rudder.quality_threshold = 0.8
        self.rudder.clip_value = 0.5
        self.lr = 1e-4
        self.weight_dec = 1e-6
        self.rudder.optimizer = torch.optim.Adam(self.rudder.net.parameters(), lr=self.lr, weight_decay=self.weight_dec)
        self.epochs = 3
        self.model_type = "stdLSTm"
        if self.use_widi_lstm:
            self.model_type = "widiLSTM"
        if self.use_gru:
            self.model_type = "GRU"
        if self.rudder.use_transformer:
            self.model_type = "transformerEncoder"
        if self.rudder.transfo_upgrade:
            self.model_type = "transformerEncoder_UP"
        if self.use_widi_uninit:
            self.model_type = "widi_unInit"
        print("using ", self.device,self.model_type)

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
    #     return quality,

    def calc_mean_of_every_n_items(self, lst, n):
        return np.mean(np.array(lst).reshape(-1, n), axis=1)

    def plot(self, returns, train_losses, test_losses):
        plt.title(
            "aO " + str(self.action_only) + " " + self.model_type + " aux loss " + str(
                self.rudder.aux_loss_multiplier) + " return mean {:.2f} lr {} w_dec {} epochs {}".format(
                np.mean(returns), self.lr, self.weight_dec,
                self.epochs))
        returns = np.array(returns)
        plt.ylabel("loss")
        plt.xlabel("episodes")
        main_loss = [l[0] for l in train_losses]
        aux_loss = [l[1] for l in train_losses]

        # main_loss = [l for e in train_losses for l in e[0]]
        # aux_loss = [l for e in train_losses for l in e[1]]
        # main_loss = self.calc_mean_of_every_n_items(main_loss, 10)
        # aux_loss = self.calc_mean_of_every_n_items(aux_loss, 10)
        plt.yscale("log")
        plt.plot(main_loss, label="train main loss")
        plt.plot(aux_loss, label="train aux loss")
        main_loss = [l[0] for l in test_losses]
        aux_loss = [l[1] for l in test_losses]

        # # main_loss = [l for l in test_losses[0]]
        # # aux_loss = [l for l in test_losses[1]]
        # main_loss = [l for e in test_losses for l in e[0]]
        # aux_loss = [l for e in test_losses for l in e[1]]
        plt.plot(main_loss, label="test main loss")
        plt.plot(aux_loss, label="test aux loss")
        plt.legend(loc="upper left")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(19.2, self.out_image_height)
        plt.savefig("trainResult_" + self.model_type, dpi=100)
        # plt.show()
        plt.close()

    def random_train_test_split(self, episodes):
        random.shuffle(episodes)
        split_index = int(len(episodes) * 0.8)
        train = episodes[:split_index]
        test = episodes[split_index:]
        return train, test

    def get_losses(self, episodes, train):
        main_loss = []
        aux_loss = []
        returns = []
        for ep in episodes:
            if train:
                _, _, ep, _, raw_loss = self.rudder.train_and_set_metrics(ep)
            else:
                _, raw_loss = self.rudder.inference_and_set_metrics(ep)
            main_loss.append(raw_loss[0])
            aux_loss.append(raw_loss[1])

            returns.append(ep.returnn)
        epoch_loss = (np.mean(main_loss), np.mean(aux_loss))
        # epoch_loss = (main_loss, aux_loss)

        return epoch_loss, returns

    # def train(self):
    #     # episodes = read_pkl_files(False)
    #     episodes = self.load_generated_demos()
    #     get_return_mean(episodes)
    #     episodes = episodes[:20]
    #     train, test = self.random_train_test_split(episodes)
    #     train_losses = []
    #     test_losses = []
    #     returns = []
    #
    #     for i in range(self.epochs):
    #         # training
    #         print(i, datetime.datetime.now().time())
    #         epoch_loss, returns_ = self.get_losses(train, True)
    #         returns.extend(returns_)
    #         train_losses.append(epoch_loss)
    #         print("train loss", epoch_loss)
    #
    #         # evaluating
    #         epoch_loss, returns_ = self.get_losses(test, False)
    #         returns.extend(returns_)
    #         # train_losses.append(epoch_loss)
    #         print("test loss", epoch_loss)
    #         test_losses.append(epoch_loss)
    #         # fname = "MyModel" + str(i) + ".pt"
    #         # torch.save(self.rudder.net.state_dict(), fname)
    #         # self.rudder.net.load_state_dict(torch.load(fname))
    #
    #     torch.save(self.rudder.net.state_dict(), self.model_type + "_model.pt")
    #     self.plot(returns, train_losses, test_losses)

    def train_one_batch(self, path, training, generated_demos=True):
        # load training files on after the other
        if generated_demos:
            episodes = self.load_generated_demos(path)
        else:
            episodes = self.load_my_demos(path)
        random.shuffle(episodes)
        print("episodes in batch", len(episodes))
        epoch_loss, returns_ = self.get_losses(episodes, training)
        del episodes
        return epoch_loss, returns_

    def iterate_files(self, path, training, generated_demos=True):
        epoch_losses = []
        returns = []

        files = os.listdir(path)
        for file in files:
            if os.path.isfile(path + file):
                batch_loss, returns_ = self.train_one_batch(path + file, training, generated_demos)
                epoch_losses.append(batch_loss)
                returns.extend(returns_)
        return epoch_losses, returns

    def calculate_rewards_from_length(self, lens, max_len):
        rewards = [1 - 0.9 * (l / max_len) if l != max_len else 0 for l in lens]
        return rewards

    def train_file_based(self, path_start, generated_demos=True):
        self.calc_and_set_mean_and_stddev_from_episode_lens(path_start+"train/")
        train_losses = []
        test_losses = []
        returns = []

        for i in range(self.epochs):
            print(i, datetime.datetime.now().time())
            path = path_start + "train/"
            batch_losses, returns_ = self.iterate_files(path, training=True, generated_demos=generated_demos)
            returns.extend(returns_)
            train_losses.append([sum(y) / len(y) for y in zip(*batch_losses)])
            print("train loss", train_losses[-1])
            # train_losses.extend(batch_losses)
            # print("train loss main", train_losses[-1][0])
            # print("train loss aux", train_losses[-1][1])
            # print("returns", returns[i * len(returns_):(i + 1) * len(returns_)])

            path = path_start + "validate/"
            batch_losses, returns_ = self.iterate_files(path, training=False, generated_demos=generated_demos)
            # returns.extend(returns_)
            test_losses.append([sum(y) / len(y) for y in zip(*batch_losses)])
            # test_losses.extend(batch_losses)

        torch.save(self.rudder.net.state_dict(), self.model_type + "_model.pt")
        self.plot(returns, train_losses, test_losses)
        self.rudder.plot_maximimum_prediction(self.model_type)
        self.rudder.plot_reduced_loss(self.model_type)

        # load test files on after the other

    def plot_reward_redistribution(self, orig_rews, redistributed_rews, actions, ax, i, label, plot_orig):
        action_dict = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up", 4: "drop", 5: "toggle",
                       6: "done"}
        actions = [action_dict[a.item()] for a in actions]
        redistributed_rews = redistributed_rews.cpu().squeeze().numpy()
        if plot_orig:
            rews = orig_rews.cpu().clone().numpy()
            rews = self.rudder.scale_rewards(rews, self.rudder.minus_to_one_scale)
            ax.plot(rews, label="original rewards")
        ax.plot(redistributed_rews, label="redistributed rewards " + str(label))
        ax.set_xticks(list(range(len(actions))))
        ax.set_xticklabels(actions, rotation=90)
        ax.legend(loc="upper right")
        ax.stem([i], [redistributed_rews[i]], linefmt="r--", markerfmt="r")
        ax.get_xticklabels()[i].set_color("red")
        return actions[i]
        # plt.show()

    def load_correct_network_parameters(self, path, file_name):
        self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128,
                              action_space=7, device=self.device,
                              use_widi=False, action_only=self.action_only).to(self.device)
        if "unInit" in file_name:
            self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128,
                                  action_space=7, device=self.device,
                                  use_widi=False, action_only=self.action_only,use_unit_widi=True).to(self.device)

        elif "widi" in file_name and "unInit" not in file_name:
            self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128,
                                  action_space=7, device=self.device,
                                  use_widi=True, action_only=self.action_only).to(self.device)

        elif "GRU" in file_name:
            self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128,
                                  action_space=7, device=self.device,
                                  use_widi=False, use_gru=True, action_only=self.action_only).to(self.device)

        elif "trans" in file_name:
            self.rudder.net.use_transformer = True
            if "UP" in file_name:
                self.rudder.net = Net(image_dim=self.image_dim, instr_dim=self.instr_dim, ac_embed_dim=128,
                                      action_space=7, device=self.device,
                                      use_widi=False, action_only=self.action_only, transfo_upgrade=True).to(
                    self.device)
        self.rudder.net.load_state_dict(torch.load(path + file_name))

    def get_predictions_from_different_models(self, model_path, short_episode):
        # path = "256Img/"
        path = model_path
        files = os.listdir(path)
        ret = []
        for model_file_name in files:
            print(model_file_name)
            self.load_correct_network_parameters(path, model_file_name)
            with torch.no_grad():
                loss, returns, quality, predictions, (main_loss, aux_loss) = self.rudder.feed_network(short_episode)
            # tmp = torch.zeros_like(predictions)
            # diff = predictions[:, 1:] - predictions[:, :-1]
            # tmp[:, 1:] = diff
            # predictions = tmp
            ret.append((predictions.squeeze(), model_file_name[:-3], (loss, main_loss, aux_loss)))
        return ret

    def evaluate_one_episode(self, start, stop, path_start, model_path, short_episode, env, top_titel=None):
        model_predictions = self.get_predictions_from_different_models(model_path, short_episode)
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
            f, axarr = plt.subplots(2, 1, figsize=(19.20, self.out_image_height))

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically

            # renderer = env.render("human")
            r = env.get_obs_render(image.cpu().numpy(), 128)
            # predictions = predictions.squeeze()
            plot_orig = True
            for el in model_predictions:
                print("loss, main, aux", el[2])
                action = self.plot_reward_redistribution(short_episode.rewards, el[0], short_episode.actions,
                                                         axarr[0], i, el[1], plot_orig)
                plot_orig = False
                # time.sleep(0.5)

            fname = "myPics/testImag" + str(i)
            r.toImage().save(fname, "PNG")

            del r
            arr = mpimg.imread(fname)
            axarr[1].imshow(arr)
            axarr[1].title.set_text("next action: " + str(action))
            combined_title = command
            if top_titel:
                combined_title += " " + top_titel
            axarr[0].title.set_text(combined_title)
            os.remove(fname)
            plt.tight_layout()
            path = "myPics/" + path_start + str(start) + "-" + str(stop) + "/"
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.savefig(path + "coolPic" + str(self.total_pics), dpi=100)
            self.total_pics += 1
            plt.close()
            # plt.gcf().close()
            # plt.show()
            # r=r.scaledToHeight(256)
        env.close()
        return

    def create_ranged_episode(self, episode, upper_limit):
        new_episode = ProcessData()
        new_episode.rewards = episode.rewards[:upper_limit]
        new_episode.actions = episode.actions[:upper_limit]
        new_episode.images = episode.images[:upper_limit]
        new_episode.instructions = episode.instructions[:upper_limit]
        new_episode.dones = episode.dones[:upper_limit]
        assert len(episode.embeddings) == 1
        assert len(episode.values) == 1
        new_episode.embeddings = episode.embeddings
        new_episode.values = episode.values
        new_episode.mission = episode.mission
        return new_episode

    def create_partial_episodes_from_failed_episode(self, episode, parts=4):
        episode: ProcessData
        ret = []
        assert len(episode.rewards) == 128
        for i in range(1, parts):
            upper_limit = int(i * (128 / parts))
            new_episode = self.create_ranged_episode(episode, upper_limit)
            ret.append(new_episode)
        ret.append(episode)
        return ret

    def visualize_low_and_high_loss_episodes(self, path_to_train_episodes,path_start, model_path, amount):
        self.calc_and_set_mean_and_stddev_from_episode_lens(path_to_train_episodes)

        low_loss, high_loss = self.get_low_and_high_loss_episode(model_path, amount)
        env = gym.make("BabyAI-PutNextLocal-v0")
        for e in high_loss:
            loss_str = "loss {:.4f} main {:.4f} aux {:.4f}".format(float(e[2]), float(e[3]), float(e[4]))
            self.evaluate_one_episode("high", "loss", path_start, model_path, e[-1], env, loss_str)
        for e in low_loss:
            loss_str = "loss {:.4f} main {:.4f} aux {:.4f}".format(float(e[2]), float(e[3]), float(e[4]))
            self.evaluate_one_episode("low", "loss", path_start, model_path, e[-1], env, loss_str)

    def visualize_failed_episode_in_parts(self, start, stop, path_start, model_path):
        print(start, stop)
        env = gym.make("BabyAI-PutNextLocal-v0")
        # self.rudder.net.load_state_dict(torch.load("MyModel.pt"))
        short_episode = self.get_random_episode_from_range(start, stop)
        episodes = self.create_partial_episodes_from_failed_episode(short_episode)
        new_start = 0
        for e in episodes:
            new_stop = len(e.dones)
            self.evaluate_one_episode(new_start, new_stop, path_start, model_path, e, env)

    def get_low_and_high_loss_episode(self, model_path, amount):
        episodes = read_pkl_files(True, "../scripts/demos/train/")
        episode_data = []
        for ep in episodes:
            model_results = self.get_predictions_from_different_models(model_path,
                                                                       ep)
            for tup in model_results:
                predictions, file_name, (loss, main_loss, aux_loss) = tup
                episode_data.append([predictions, file_name, loss, main_loss, aux_loss, ep])
                print("loss, len", loss.item(), len(predictions))
        episode_data.sort(key=lambda x: x[2])
        return episode_data[:amount], episode_data[-amount:]

    def get_random_episode_from_range(self, start, stop):
        episodes = read_pkl_files(True)
        random.shuffle(episodes)
        short_episode = None
        fist = False
        for e in episodes:
            e: ProcessData
            if start < len(e.dones) < stop:
                short_episode = e
                break
        assert short_episode is not None
        return short_episode

    def evaluate(self, start, stop, path_start, model_path):
        print(start, stop)
        env = gym.make("BabyAI-PutNextLocal-v0")
        # self.rudder.net.load_state_dict(torch.load("MyModel.pt"))
        short_episode = self.get_random_episode_from_range(start, stop)
        self.evaluate_one_episode(start, stop, path_start, model_path, short_episode, env)

    def load_generated_demos(self, path, max_steps=128):
        with open(path, "rb") as f:
            vocab = {"put": 1, "the": 2, "grey": 3, "key": 4, "next": 5, "to": 6, "red": 7,
                     "box": 8, "yellow": 9, "blue": 10, "green": 11, "purple": 12, "ball": 13}
            episodes = pickle.load(f)
        demos = transform_demos(episodes)
        device = "cuda"
        transformed = []
        for demo in demos:
            p = ProcessData()
            p.mission = demo[0][0]["mission"].lower()
            step_count = len(demo)
            reward = 1 - 0.9 * (step_count / max_steps)
            assert 0 <= reward < 1
            if step_count == max_steps:
                reward = 0
            p.rewards = torch.zeros(len(demo))
            p.rewards[-1] = reward
            p.dones = [el[2] for el in demo]

            images = np.array([action[0]["image"] for action in demo])
            images = torch.tensor(images, device=device, dtype=torch.float)
            p.images = images

            p.actions = torch.tensor([action[1] for action in demo], device="cuda", dtype=torch.long)
            tokens = re.findall("([a-z]+)", demo[0][0]["mission"].lower())
            assert p.mission.split() == tokens
            p.instructions = torch.from_numpy(np.array([vocab[token] for token in tokens])).repeat(len(demo), 1)
            # dummy elements
            p.embeddings = torch.zeros(1)
            p.values = torch.zeros(1)
            assert len(p.dones) == len(p.rewards) == len(p.actions) == len(p.instructions) == len(p.images) == len(demo)
            transformed.append(p)
        return transformed

    def load_my_demos(self, path):
        with open(path, "rb") as f:
            episodes = pickle.load(f)
        return episodes

    def calc_and_set_mean_and_stddev_from_episode_lens(self, path):
        files = [f for f in os.listdir(path) if os.path.isfile(path + f)]
        lens = []
        total = 0
        for file in files:
            with open(path + file, "rb") as f:
                episodes = pickle.load(f)
                total += len(episodes)
                [lens.append(len(e[2])) for e in episodes]
        assert len(lens) >= 1
        c = Counter(lens)
        print(c)
        max_len = 128
        rewards = self.calculate_rewards_from_length(lens, max_len)
        self.rudder.mean = np.mean(rewards)
        self.rudder.std_dev = np.std(rewards)

    def calc_rew_of_generated_episodes(self, path):
        rews = []
        lens = []
        total = 0
        pos_rets = 0
        for file in os.listdir(path):
            print(file)
            episodes = self.load_generated_demos(path + file)
            for episode in episodes:
                rew = episode.rewards[-1].item()
                rews.append(rew)
                lens.append(len(episode.actions))
                if rew > 0:
                    pos_rets += 1
                total += 1
            del episodes
            torch.cuda.empty_cache()

        print(np.mean(rews))
        print("total eps", total)
        print("pos ret", pos_rets)
        print("percent of susccesfull eps", pos_rets / total)
        print("lens")
        print(Counter(lens).most_common(20))


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


def read_pkl_files(evaluate, path=None):
    training2 = Training()
    all_episodes = []
    if not path:
        if evaluate:
            path = "../scripts/demos/validate/"
            # path = "../scripts/replays4/"
        else:
            path = "../scripts/replays3/"

    files = os.listdir(path)
    limit = int(len(files) * 0.8)
    for file in files:
        if os.path.isfile(path + file):
            with open(path + file, "rb") as f:
                # episodes = pickle.load(f)
                episodes = training2.load_generated_demos(path + file)[:1000]
                all_episodes.extend(episodes)
    print("episodes", len(all_episodes))
    return all_episodes


def get_return_mean(episodes):
    rets = []
    for episode in episodes:
        rets.append(torch.sum(episode.rewards).item())
    print("mean return", np.mean(rets))


def do_multiple_evaluations(model_path, parent_folder):
    ranges = [(0, 12), (12, 20), (20, 40), (40, 60), (60, 128), (127, 129)]
    runs = 3
    for i in range(runs):
        path = parent_folder + "run" + str(i) + "/"
        # Path(path).mkdir(parents=True, exist_ok=True)
        for r in ranges:
            training.evaluate(r[0], r[1], path, model_path)


def save_cleaned_episodes(wrote_files, duplicate_free):
    with open("cleaned_eps3/f" + str(wrote_files) + ".pkl", "wb") as f:
        pickle.dump(duplicate_free, f)


def find_unique_episodes(path):
    files = os.listdir(path)
    seen = Counter()
    total_episodes = 0
    duplicates = 0
    duplicate_free = []
    wrote_files = 0
    for file in files:
        with open(path + file, "rb") as f:
            episodes = pickle.load(f)
            for episode in episodes:
                # triple = (episode.mission, "".join([str(a.item()) for a in episode.actions]), str(episode.rewards[-1].item()))
                triple = episode.mission + "{0:.3f}".format(episode.rewards[-1].item()) + "-" + "".join(
                    [str(a.item()) for a in episode.actions])

                # print(seen.most_common(2))
                if triple in seen:
                    # print(seen.most_common(2))
                    duplicates += 1
                else:
                    duplicate_free.append(episode)
                if len(duplicate_free) % 10000 == 0:
                    save_cleaned_episodes(wrote_files, duplicate_free)
                    duplicate_free = []
                    wrote_files += 1

                seen.update([triple])
                total_episodes += 1
    save_cleaned_episodes(wrote_files, duplicate_free)

    print("from {} episodes we have {} percent duplicates".format(total_episodes, duplicates / total_episodes))
    for l in seen.most_common(10):
        print(l)


def calc_memory_saving_ret_mean(path):
    files = os.listdir(path)
    rews = []
    for file in files:
        with open(path + file, "rb") as f:
            episodes = pickle.load(f)
            rews.extend([episode.rewards[-1].item() for episode in episodes])
    print(np.mean(rews))


def extract_positive_return_episodes(src_path, dest_path):
    files = os.listdir(src_path)
    pos_rets = []
    for file in files:
        with open(src_path + file, "rb") as f:
            episodes = pickle.load(f)
            for episode in episodes:
                if episode.rewards[-1].item() > 0:
                    pos_rets.append(episode)
    with open(dest_path + "pos_rets.pkl", "wb") as f:
        pickle.dump(pos_rets, f)


def plot_rewards(rewards):
    mean_rew = np.mean(rewards)

    w = Counter(rewards)
    plt.bar(w.keys(), w.values(), 0.01)
    plt.xlabel("return of episode")
    plt.ylabel("count")
    plt.yscale('log')
    plt.title("mean return {:.2f}".format(mean_rew) +
              " std: {:.2f}".format(np.std(rewards)) + " var: {:.2f}".format(np.var(rewards)))
    plt.show()


def create_episode_len_histogram(path):
    files = [f for f in os.listdir(path) if os.path.isfile(path + f)]
    lens = []
    total = 0
    for file in files:
        with open(path + file, "rb") as f:
            episodes = pickle.load(f)
            total += len(episodes)
            [lens.append(len(e[2])) for e in episodes]
    c = Counter(lens)
    print(c)
    max_len = 128
    rewards = training.calculate_rewards_from_length(lens, max_len)
    mean_rew = np.mean(rewards)
    plt.xlabel("episode length")
    plt.ylabel("count")
    plt.yscale('log')
    plt.title("mean return {:.2f}".format(mean_rew) + " episodes: " + str(total) + " failed: " + str(c[max_len]))
    plt.bar(c.keys(), c.values())
    plt.show()

    rewards = np.array(rewards)
    plot_rewards(rewards)

    # rewards = rewards*1.84-0.92
    # rewards = rewards*2.42-1.1
    rewards = (rewards - np.mean(rewards)) / np.std(rewards)

    plot_rewards(rewards)


# dir2 = "/home/nick/Downloads/trainset1M/orig/"
# create_episode_len_histogram("../scripts/demos/train/")
# create_episode_len_histogram(dir2)
# env = gym.make("BabyAI-PutNextLocal-v0")
# sys.settrace
training = Training()
# training.visualize_low_and_high_loss_episodes("../scripts/demos/train/"
#                                               ,"lossVisualizedMinus1Plus1LSTMandGRUWidi/","models/",6)
# training.visualize_failed_episode_in_parts(127, 129, "failedVisualized1Million0.5Aux1e-6LRNoAuxTime/",
#                                            "1Million0.5Aux1e-6LRNoAuxTime/", )
# training.calc_rew_of_generated_episodes("../scripts/demos/train/")
# do_multiple_evaluations("models/1Million0.5Aux1e-5LRNoAuxTime/", "EVAL_GRU_1Million0.5Aux1e-5LRNoAuxTime/")
training.train_file_based("../scripts/demos/")
# find_unique_episodes("../scripts/replays7/")
# calc_memory_saving_ret_mean("../scripts/demos/train/")
# my_path = "testi/"
# extract_positive_return_episodes(my_path,my_path)

# check format of old saving format
