from collections import Counter

import numpy as np
import torch
from myScripts.ReplayBuffer import ProcessData
from rudder.rudder_evaluate import build_loss_str

from rudder.rudder_imitation_learning import RudderImitation
from rudder.rudder_plot import RudderPlotter
from rudder.rudder_replay_buffer import RudderReplayBuffer
from torch.optim import Adam


class NonParsedDummyArgs:
    def __init__(self, instr_dim, memory_dim, image_dim, lr):
        self.model = "rudderRLModel"
        self.no_instr = False
        self.no_mem = False
        self.instr_arch = "gru"
        self.arch = 'expert_filmcnn'
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.env = "BabyAI-PutNextLocal-v0"
        self.log_interval = 1
        self.lr = lr
        self.optim_eps = 1e-5
        self.recurrence = 20


class Rudder:
    def __init__(self, nr_procs, device, frames_per_proc, instr_dim, memory_dim, image_dim, lr, base_rl_algo):

        self.max_grad_norm = 0.5
        self.mu = 1
        self.quality_threshold = 0.8
        dummy_args = NonParsedDummyArgs(instr_dim, memory_dim, image_dim, lr)
        self.il_learn = RudderImitation(None, True, True, dummy_args)
        # these 2 must be updated when replaybuffer full and then after every new insert
        self.il_learn.mean = 0
        self.il_learn.std_dev = 1
        self.nr_procs = nr_procs
        self.replay_buffer = RudderReplayBuffer(nr_procs, frames_per_proc, device)
        self.first_training_done = False
        self.current_quality = 0
        self.grad_norms = []
        self.grad_norm = -1
        self.device = device
        self.frames_per_proc = frames_per_proc
        self.base_rl_algo = base_rl_algo
        self.lr = lr
        self.optimizer = Adam(lr=self.lr, params=self.il_learn.acmodel.parameters())
        self.action_dict = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up", 4: "drop", 5: "toggle",
                            6: "done"}

    def get_process_data(self, index, obs, *args):
        return [a.transpose(0, 1)[index] for a in args] + [list(map(list, zip(*obs)))[index]]

    def minus_one_to_one_scale(self, rewards, buffer_rewards=None):
        if buffer_rewards is not None:
            tmp_rewards = torch.cat((rewards.transpose(0, 1), buffer_rewards), dim=0)
        else:
            tmp_rewards = rewards
        std_dev = torch.std(tmp_rewards)
        if std_dev != 0:
            rewards = (rewards - torch.mean(tmp_rewards)) / std_dev
        return rewards

    # def fill_buffer(self, masks, rewards, values, actions, obs, dones):
    #     # # rewards to zero mean unit variance
    #     # rewards = rewards / 20
    #     # if self.replay_buffer.added_episodes > 0:
    #     #     rewards = self.minus_one_to_one_scale(rewards, self.replay_buffer.rewards)
    #     # else:
    #     #     rewards = self.minus_one_to_one_scale(rewards)
    #     print("val max min", values.max().item(), values.min().item())
    #     for i in range(self.nr_procs):
    #         masks_, rewards_, values_, actions_, dones_, obs_, = self.get_process_data(i, obs, masks, rewards, values,
    #                                                                                    actions, dones)
    #         final_loss, seq_return, _ = self.get_loss_for_sequence(obs_, masks_, rewards_, actions_, values_, dones_)
    #
    #         if not self.replay_buffer.buffer_full():
    #             idx = i
    #         else:
    #             idx = self.replay_buffer.new_get_replacement_index(final_loss.item(), seq_return.item())
    #         if idx != -1:
    #             self.replay_buffer.add_single_sequence(masks_, rewards_, values_, actions_, obs_, seq_return,
    #                                                    final_loss, dones_, idx)

    def fill_buffer_batch(self, masks, rewards, values, actions, obs, dones, update,model_name):
        # # rewards to zero mean unit variance
        # rewards = rewards / 20
        # if self.replay_buffer.added_episodes > 0:
        #     rewards = self.minus_one_to_one_scale(rewards, self.replay_buffer.rewards)
        # else:
        #     rewards = self.minus_one_to_one_scale(rewards)
        print("val max min", values.max().item(), values.min().item())
        my_obs, my_actions, my_masks, my_rewards, my_values, my_dones = self.get_data_for_every_process(obs, masks,
                                                                                                        rewards, values,
                                                                                                        actions, dones)
        loss, seq_return, (aux, main, predictions, quality) = self.get_loss_for_batch(my_obs, my_masks, my_rewards,
                                                                                      my_actions, my_values, my_dones,
                                                                                      is_training=False)
        if update % 200 == 0:
            self.visualize_current_reward_redistribution(loss, my_obs, my_actions, my_rewards, aux, main, predictions,
                                                         update, my_dones,model_name)
        for i in range(my_actions.shape[0]):
            masks_, rewards_, values_, actions_, obs_, seq_return_, final_loss, dones_ = my_masks[i], my_rewards[i], \
                                                                                         my_values[i], my_actions[i], \
                                                                                         my_obs[i], seq_return[i], loss[
                                                                                             i], my_dones[i]
            if not self.replay_buffer.buffer_full():
                idx = i
            else:
                idx = self.replay_buffer.new_get_replacement_index(final_loss.item(), seq_return_.item())
            if idx != -1:
                self.replay_buffer.add_single_sequence(masks_, rewards_, values_, actions_, obs_, seq_return_,
                                                       final_loss, dones_, idx)

    def flip_zeros_and_ones(self, tensor):
        # torch.where(tensor == 0, torch.ones(1, device=self.device), torch.zeros(1, device=self.device))
        return (tensor - 1) * (-1)

    def net_single_step_feed_forward(self, memory, memories, preprocessed_obs, mask, action, step):
        model_results = self.il_learn.acmodel(preprocessed_obs, memory * mask.unsqueeze(1), actions=action)
        new_memory = model_results["memory"]
        memories[step] = memory
        memory = new_memory
        return memory, memories, model_results

    def calc_quality(self, diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality = 1 - (np.abs(diff.item()) / self.mu) * 1 / (1 - self.quality_threshold)
        return quality

    def calc_quality_batch(self, diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality = 1 - (torch.abs(diff) / self.mu) * 1 / (1 - self.quality_threshold)
        return quality

    def calculate_batch_loss(self, rewards, repeated_rewards, done, predictions):
        # dont use the standard mean for main loss because many zeros in batch will decrease it
        # take only the return (== reward in this environment) into account
        diff = ((rewards.detach().clone() - predictions.detach().clone()) * done).sum(dim=1) / torch.sum(done, dim=-1)
        quality = self.calc_quality_batch(diff)
        assert (torch.sum(done, dim=-1) > 0).all()

        main_loss = (((rewards - predictions) * done) ** 2).sum(dim=1) / torch.sum(done, dim=-1)
        aux_loss = ((repeated_rewards - predictions) ** 2).mean(dim=1)
        final_loss = main_loss + self.il_learn.aux_loss_multiplier * aux_loss
        return final_loss, (main_loss.detach().clone(), aux_loss.detach().clone(), quality)

    def calculate_online_loss(self, rewards, repeated_rewards, done, predictions):
        # dont use the standard mean for main loss because many zeros in batch will decrease it
        # take only the return (== reward in this environment) into account
        diff = ((rewards.detach().clone() - predictions.detach().clone()) * done).sum() / torch.sum(done, dim=-1)
        quality = self.calc_quality(diff)
        assert torch.sum(done, dim=-1) > 0
        main_loss = (((rewards - predictions) * done) ** 2).sum() / torch.sum(done, dim=-1)
        aux_loss = ((repeated_rewards - predictions) ** 2).mean()
        final_loss = main_loss + self.il_learn.aux_loss_multiplier * aux_loss
        return final_loss, (main_loss.detach().clone(), aux_loss.detach().clone(), quality)

    def feed_single_sequence_to_net(self, obss, actions, masks, is_training=False, batch=False):
        if batch:
            memories = torch.zeros([actions.shape[1], actions.shape[0], self.il_learn.acmodel.memory_size],
                                   device=self.device)
            memory = torch.zeros(actions.shape[0], self.il_learn.acmodel.memory_size, device=self.device)
            actions = actions.to(dtype=torch.long)

        else:
            memories = torch.zeros([len(actions), self.il_learn.acmodel.memory_size], device=self.device)
            memory = torch.zeros(self.il_learn.acmodel.memory_size, device=self.device).unsqueeze(0)
            actions = torch.tensor([action.to(dtype=torch.long) for action in actions], device=self.device)

        predictions = []

        if is_training:
            iterations = self.base_rl_algo.recurrence
        else:
            iterations = self.frames_per_proc

        for i in range(iterations):
            # obs net to be a list because preprocess_obss() expects batch input
            if not batch:
                obs = [obss[i]]
                action = actions[i].unsqueeze(0)
                mask = masks[i].unsqueeze(0)
            else:
                obs = obss[:, i]
                action = actions[:, i]
                mask = masks[:, i]

            preprocessed_obs = self.base_rl_algo.preprocess_obss(obs, device=self.device)
            if is_training:
                memory, memories, model_results = self.net_single_step_feed_forward(memory, memories, preprocessed_obs,
                                                                                    mask, action, i)
            else:
                with torch.no_grad():
                    memory, memories, model_results = self.net_single_step_feed_forward(memory, memories,
                                                                                        preprocessed_obs, mask, action,
                                                                                        i)
            predictions.append(model_results["value"])

        if batch:
            return torch.stack(predictions, dim=1)

        return torch.cat(predictions, dim=-1)

    def info_print(self, idx, returnn, loss, main, aux, predictions, rewards,done):
        diff_at_done = ((rewards[:len(predictions)] - predictions)**2)[done[:len(predictions)] == 1]
        diff_at_done = ["{:.2f}".format(e.item()) for e in diff_at_done ]
        print(
            "sample {} return {:.2f} loss {:.4f}"
            " mainL {:.4f}  auxL {:.4f} predMax {:.2f} rewMax {:.2f} diffAtDone {}".format(idx, returnn.item(), loss.item(),
                                                                             main.item(),
                                                                             aux.item(), predictions[0].max().item(),
                                                                             rewards.max().item(),diff_at_done))

    def get_batch_data(self):
        my_obs = []
        my_masks = []
        my_rewards = []
        my_actions = []
        my_values = []
        my_dones = []
        episodes, ids = self.replay_buffer.sample_episodes()
        for i, episode in enumerate(episodes):
            obs, masks, rewards, actions, values, dones = episode
            my_obs.append(obs)
            my_masks.append(masks)
            my_rewards.append(rewards)
            my_actions.append(actions)
            my_values.append(values)
            my_dones.append(dones)
        my_obs = np.stack(my_obs)
        my_masks = torch.stack(my_masks)
        my_rewards = torch.stack(my_rewards)
        my_actions = torch.stack(my_actions)
        my_values = torch.stack(my_values)
        my_dones = torch.stack(my_dones)
        return my_obs, my_masks, my_rewards, my_actions, my_values, my_dones, ids

    def train_on_buffer_data(self):
        bad_quality = True
        losses = []
        aux_losses = []
        while bad_quality:
            qualities_bools = set()
            qualities = []
            for _ in range(5):
                my_obs, my_masks, my_rewards, my_actions, my_values, my_dones, ids = self.get_batch_data()
                loss, seq_return, (aux, main, predictions, quality) = self.get_loss_for_batch(my_obs, my_masks,
                                                                                              my_rewards,
                                                                                              my_actions, my_values,
                                                                                              my_dones, True)
                for i, idx in enumerate(ids):
                    self.replay_buffer.losses[ids[i]] = loss[i].detach().clone().item()
                    qualities_bools.add(quality[i].item() > 0)
                    qualities.append(np.clip(quality[i].item(), 0.0, 1.0))

                self.optimizer.zero_grad()
                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.il_learn.acmodel.parameters(), self.max_grad_norm)
                grad_norm = sum(
                    p.grad.data.norm(2) ** 2 for p in self.il_learn.acmodel.parameters() if p.grad is not None) ** 0.5
                self.optimizer.step()
                losses.append(loss.mean().item())
                aux_losses.append(aux.mean().item())

            self.current_quality = np.mean(qualities)
            if False not in qualities_bools:
                bad_quality = False
            self.info_print(ids[i], seq_return[i], loss[i], main[i], aux[i], predictions[i], my_rewards[i],my_dones[i])
        return np.mean(losses),np.mean(aux_losses), grad_norm

    def redistribute_reward(self, predictions, rewards):
        # Use the differences of predictions as redistributed reward
        redistributed_reward = predictions[:, 1:] - predictions[:, :-1]

        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
        redistributed_reward = torch.cat([predictions[:, :1], redistributed_reward], dim=1)
        returns = rewards.sum(dim=1)
        predicted_returns = redistributed_reward.sum(dim=1)
        prediction_error = returns - predicted_returns

        # Distribute correction for prediction error equally over all sequence positions
        redistributed_reward += prediction_error[:, None] / redistributed_reward.shape[1]
        return redistributed_reward

    def predict_new_rewards(self, obs, masks, rewards, values, actions, dones):
        # rewards = rewards / 20
        out_rewards = []
        best_actions = []
        for i in range(self.nr_procs):
            masks_, rewards_, values_, actions_, dones_, obs_ = self.get_process_data(i, obs, masks, rewards, values,
                                                                                      actions, dones)
            predictions = self.feed_single_sequence_to_net(obs_, actions_, masks_)
            redistributed_reward = self.redistribute_reward(predictions.unsqueeze(0), rewards_.unsqueeze(0))
            out_rewards.append(redistributed_reward.squeeze(0))
            best_actions.append(actions_[torch.argmax(redistributed_reward)].item())
        out_rewards = torch.stack(out_rewards)
        count = Counter(best_actions)
        best_action_strings = [self.action_dict[a[0]] for a in count.most_common(3)]
        print("best actions", best_action_strings)

        return out_rewards.transpose(0, 1)

    def get_data_for_every_process(self, obs, masks, rewards, values, actions, dones):
        out_rewards = []
        best_actions = []
        my_obs = []
        my_masks = []
        my_rewards = []
        my_actions = []
        my_values = []
        my_dones = []
        for i in range(self.nr_procs):
            masks_, rewards_, values_, actions_, dones_, obs_ = self.get_process_data(i, obs, masks, rewards, values,
                                                                                      actions, dones)
            my_obs.append(obs_)
            my_masks.append(masks_)
            my_rewards.append(rewards_)
            my_actions.append(actions_)
            my_values.append(values_)
            my_dones.append(dones_)
        my_obs = np.stack(my_obs)
        my_masks = torch.stack(my_masks)
        my_rewards = torch.stack(my_rewards)
        my_actions = torch.stack(my_actions)
        my_values = torch.stack(my_values)
        my_dones = torch.stack(my_dones)
        return my_obs, my_actions, my_masks, my_rewards, my_values, my_dones

    def visualize_current_reward_redistribution(self, loss, my_obs, my_actions, my_rewards, aux, main, all_predictions,
                                                update, dones,model_name_orig):
        for i in range(len(loss))[:10]:
            orig_rewards = my_rewards[i]
            predictions = all_predictions[i]
            model_name = model_name_orig+"_" + str(update)
            final_loss, main_loss, aux_loss = loss[i], main[i], aux[i]
            obs = my_obs[i]
            actions = my_actions[i]
            episode = ProcessData()
            done = dones[i]
            episode.rewards = orig_rewards
            episode.instructions = [e["mission"] for e in obs]
            episode.images = torch.tensor([e["image"] for e in obs], device=self.il_learn.device)
            episode.actions = actions
            loss_str = build_loss_str((final_loss, main_loss, aux_loss, predictions, orig_rewards, model_name))
            rudder_plotter = RudderPlotter(None)
            output_path_prefix = "newEval/"
            # model pred contains (predictions.squeeze(), model_file_name[:-3], (loss, main_loss, aux_loss))
            model_predictions = [(predictions, model_name, (final_loss, (main_loss, aux_loss)))]
            rudder_plotter.plot_reward_redistribution(str(torch.sum(episode.rewards).item()),
                                                      str(torch.sum(done).item()) + "_" + str(i),
                                                      output_path_prefix + model_name + "_Eval/",
                                                      model_predictions, episode,
                                                      self.il_learn.env, top_titel=loss_str, multi_commands=True)

    def predict_new_rewards_batch(self, obs, masks, rewards, values, actions, dones):
        # rewards = rewards / 20
        my_obs, my_actions, my_masks, my_rewards, _, _ = self.get_data_for_every_process(obs, masks, rewards, values,
                                                                                         actions, dones)
        predictions = self.feed_single_sequence_to_net(my_obs, my_actions, my_masks, batch=True)
        redistributed_reward = self.redistribute_reward(predictions, my_rewards)
        # out_rewards.append(redistributed_reward.squeeze(0))
        best_actions = [my_actions[i][torch.argmax(line)].item() for i, line in enumerate(redistributed_reward)]
        out_rewards = redistributed_reward
        count = Counter(best_actions)
        best_action_strings = [self.action_dict[a[0]] for a in count.most_common(5)]
        print("best actions", best_action_strings)

        return out_rewards.transpose(0, 1)

    def get_loss_for_batch(self, obs, masks, rewards, actions, values, dones, is_training=False):
        predictions = self.feed_single_sequence_to_net(obs, actions, masks, is_training, True)
        rewards = rewards[:, :predictions.shape[1]].clone()
        seq_return = torch.sum(rewards, dim=1)
        values = values[:, :predictions.shape[1]]
        # dones = self.flip_zeros_and_ones(masks).unsqueeze(0)
        dones = dones[:, :predictions.shape[1]]
        dones[:, -1] = 1
        rewards[:, -1] = torch.where(rewards[:, -1] == 0, values[:, -1], rewards[:, -1])
        repeated_rewards = self.create_repeated_reward(rewards)
        final_loss, (main, aux, quality) = self.calculate_batch_loss(rewards, repeated_rewards,
                                                                     dones, predictions)
        return final_loss, seq_return, (aux, main, predictions, quality)

    def get_loss_for_sequence(self, obs, masks, rewards, actions, values, dones, is_training=False):
        rewards = rewards.unsqueeze(0)
        values = values.unsqueeze(0)
        dones = dones.unsqueeze(0)
        # overwrite missing rewards with values ( see paper appendix A 4.2.3)#
        predictions = self.feed_single_sequence_to_net(obs, actions, masks, is_training).unsqueeze(0)
        rewards = rewards[:, :predictions.shape[1]]
        seq_return = torch.sum(rewards)
        values = values[:, :predictions.shape[1]]
        # dones = self.flip_zeros_and_ones(masks).unsqueeze(0)
        dones = dones[:, :predictions.shape[1]]
        dones[:, -1] = 1
        rewards[:, -1] = torch.where(rewards[:, -1] == 0, values[:, -1], rewards[:, -1])
        repeated_rewards = self.create_repeated_reward(rewards)
        final_loss, (aux, main, quality) = self.calculate_online_loss(rewards, repeated_rewards,
                                                                      dones, predictions)
        return final_loss, seq_return, (aux, main, predictions, quality)

    # def set_losses_for_each_sequence(self):
    #     for i in range(self.nr_procs):
    #         obs, masks, rewards, actions, values = self.replay_buffer.get_single_sequence(i)
    #         final_loss, sequence_return = self.get_loss_for_sequence(obs, masks, rewards, actions, values)
    #         print(final_loss)
    #         self.replay_buffer.losses[i] = final_loss
    #         self.replay_buffer.returns[i] = sequence_return

    def create_repeated_reward(self, sparse_reward):
        # line per line
        out = []
        for i, process_rewards in enumerate(sparse_reward):
            reward_indices = torch.nonzero(process_rewards, as_tuple=False)
            repeated_line = []
            start = -1
            for rew_idx in reward_indices:
                repeated_line.append(process_rewards[rew_idx].expand(rew_idx - start))
                start = rew_idx
            if not repeated_line:
                repeated_line = process_rewards
            else:
                repeated_line = torch.cat(repeated_line)
            out.append(repeated_line)
        return torch.stack(out)

        pass
