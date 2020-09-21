import numpy as np
import torch

from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
from rudder.rudder_evaluate import calculate_my_flat_loss, transform_data_for_loss_calculation
from rudder.rudder_imitation_learning import RudderImitation


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
    def __init__(self, nr_procs, device, frames_per_proc, instr_dim, memory_dim, image_dim, lr):
        dummy_args = NonParsedDummyArgs(instr_dim, memory_dim, image_dim, lr)
        self.il_learn = RudderImitation(None, True, True, dummy_args)
        # these 2 must be updated when replaybuffer full and then after every new insert
        self.il_learn.mean = 0
        self.il_learn.std_dev = 1
        self.replay_buffer = ReplayBuffer(nr_procs, 128, device, frames_per_proc)
        self.first_training_done = False
        self.current_quality = 0
        self.grad_norms = []
        self.grad_norm = -1

    def add_timestep_data(self, debug, queue_in_rudder, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)

        self.new_add_to_replay_buffer(complete_episodes, debug)
        return

    def new_add_to_replay_buffer(self, complete_episodes, debug=False):
        for ce in complete_episodes:

            self.inference_and_set_metrics(ce)
            if self.replay_buffer.buffer_full():
                idx = self.replay_buffer.new_get_replacement_index(ce)
            else:
                idx = self.replay_buffer.added_episodes
                self.replay_buffer.added_episodes += 1
            if idx != -1:
                self.replay_buffer.new_replace_episode_data(idx, ce)

    def create_rl_compatible_demo(self, complete_episode):
        compatible_demo = []
        for i, done in enumerate(complete_episode.dones):
            obs = {'image': complete_episode.images[i].cpu().numpy(),
                   'direction': None,
                   'mission': complete_episode.mission[0]}
            compatible_demo.append((obs, complete_episode.actions[i], done))
        return compatible_demo

    def get_loss_from_finished_episode(self, finished_episode):
        predictions, orig_rewards, dones, actions, obs, model_name = finished_episode
        assert obs[0][0]["mission"] == obs[-1][0]["mission"]

        # transform to tensors
        reward_repeated_step, my_done_step, predictions, orig_rewards = \
            transform_data_for_loss_calculation(predictions, orig_rewards, dones, self.il_learn)

        # calculate loss
        final_loss, (main_loss, aux_loss), quality = calculate_my_flat_loss(predictions, orig_rewards, self.il_learn,
                                                                            reward_repeated_step, return_quality=True)

        return final_loss, (main_loss, aux_loss), quality, orig_rewards, predictions

    def inference_and_set_metrics(self, complete_episode):
        compatible_demo = self.create_rl_compatible_demo(complete_episode)
        log, finished_episode, _ = self.il_learn.run_epoch_recurrence_one_batch([(compatible_demo, None)],
                                                                                is_training=False,
                                                                                rl_input=True, rudder_eval=True)

        final_loss, (main_loss, aux_loss), quality, orig_rewards, predictions = self.get_loss_from_finished_episode(
            finished_episode[0])
        loss = final_loss.detach().item()
        returnn = torch.sum(orig_rewards, dim=-1).detach().item()
        # returnn  = np.sum(orig_rewards,axis=-1)
        complete_episode.loss = loss
        complete_episode.rewards = torch.stack(complete_episode.rewards)
        complete_episode.returnn = returnn
        raw_loss = (main_loss, aux_loss)

        return quality, raw_loss, predictions

    def train_and_set_metrics_batch(self, episodes, sample_indices):
        demos = [(self.create_rl_compatible_demo(e), (i, sample_indices[i])) for i, e in enumerate(episodes)]

        log, finished_episodes, indices = self.il_learn.run_epoch_recurrence_one_batch(demos, is_training=True,
                                                                                       rl_input=True, rudder_eval=True)
        qualities = []
        full_predictions = []
        last_timestep_prediction = []
        for finished_episode in finished_episodes:
            final_loss, (main_loss, aux_loss), quality, orig_rewards, predictions = self.get_loss_from_finished_episode(
                finished_episode)
            full_predictions.append(predictions.unsqueeze(0))
            last_timestep_prediction.append(predictions[-1].item())
            qualities.append(np.clip(quality, 0.0, 0.5))

        losses = []
        last_rewards = []
        for index in indices:
            complete_episode = episodes[index[0]]
            index_in_buffer = index[1]
            loss = final_loss.detach().item()
            returnn = torch.sum(orig_rewards, dim=-1).detach().item()
            # returnn  = np.sum(orig_rewards,axis=-1)
            complete_episode.loss = loss
            # complete_episode.rewards = torch.stack(complete_episode.rewards)
            complete_episode.returnn = returnn
            raw_loss = (main_loss, aux_loss)
            self.replay_buffer.fast_losses[index_in_buffer] = complete_episode.loss
            losses.append(complete_episode.loss)
            last_rewards.append(complete_episode.rewards[-1].item())
            self.grad_norms.append(log["grad_norm"])
        print("sample {} return {:.2f} loss {:.4f} predX {:.2f}  rewX {:.2f} pred[-1] {:.2f} rew[-1]  {:.2f} ".
              format(index_in_buffer, complete_episode.returnn, complete_episode.loss,
                     predictions.mean().item(), complete_episode.rewards.mean().item(),
                     predictions[-1].item(), complete_episode.rewards[-1].item()))
        return qualities, full_predictions, last_rewards, losses, last_timestep_prediction

    def train_full_buffer(self):
        # print("train_full_buffer")
        # self.net.train()
        losses = []
        full_predictions = []
        last_timestep_prediction = []
        last_rewards = []
        bad_quality = True
        while bad_quality:
            qualities_bools = set()
            qualities = []
            for epoch in range(5):
                episodes, episodes_ids = self.replay_buffer.sample_episodes()
                qualities_, full_predictions_, last_rewards_, losses_, last_timestep_prediction_ = self.train_and_set_metrics_batch(
                    episodes, episodes_ids)
                qualities.extend(qualities_)
                full_predictions.extend(full_predictions_)
                losses.extend(losses_)
                last_timestep_prediction.extend(last_timestep_prediction_)
                [qualities_bools.add(q > 0) for q in qualities_]
                # for i, episode in enumerate(episodes):
                #     quality, predictions, episode, grad_norm, raw_loss = self.train_and_set_metrics(episode)
                    # self.grad_norms.append(grad_norm.item())
                    # self.replay_buffer.fast_losses[episodes_ids[i]] = episode.loss
                    # full_predictions.append(predictions.unsqueeze(0))
                    # last_timestep_prediction.append(predictions[0][-1].item())
                    # losses.append(episode.loss)
                    # last_rewards.append(episode.rewards[-1].item())
                    # assert episode.returnn==self.replay_buffer.fast_returns[episodes_ids[i]]
                    # qualities_bools.add(quality > 0)
                    # qualities.append(np.clip(quality, 0.0, 0.5))
            self.current_quality = np.mean(qualities)
            # self.current_quality = 0.25
            # print("sample {} return {:.2f} loss {:.4f} predX {:.2f}  rewX {:.2f} pred[-1] {:.2f} rew[-1]  {:.2f} ".
            #       format(episodes_ids[-1], episode.returnn, episode.loss,
            #              predictions.mean().item(), episode.rewards.mean().item(),
            #              predictions[-1][-1][-1].item(), episode.rewards[-1].item()))
            if False not in qualities_bools:
                bad_quality = False
        self.grad_norm = np.mean(self.grad_norms)
        self.grad_norms = []
        # full_predictions = torch.cat(full_predictions, dim=-1)
        return np.mean(losses), np.mean(last_timestep_prediction), np.mean(
            last_rewards)  # , torch.mean(full_predictions)

    def predict_full_episode(self, episode: ProcessData):
        # predictions, hidden, pred_plus_ten_ts = self.net(episode, None, True, self.use_transformer)
        quality, raw_loss, predictions = self.inference_and_set_metrics(episode)
        return predictions,None

    def do_part_prediction(self, proc_id, timestep):
        with torch.no_grad():
            episode = self.replay_buffer.proc_data_buffer[proc_id]
            pred, _ = self.predict_full_episode(episode)
            episode_len = len(episode.dones)
            to_add = min(episode_len, timestep + 1)
            previous_pred = None
            # might be out of range because frames per proc is smaller than episode len
            try:
                previous_pred = pred.squeeze()[-(to_add + 1)]
            except:
                pass
            # single dimension
            try:
                pred = pred.squeeze()[-to_add:]
            except IndexError:
                pred = pred.squeeze().unsqueeze(0)
            # we need to get the differences to previous timestep predictions
            previous_timesteps = torch.zeros(pred.shape, device=self.device)
            # leave the first value at zero to not subtract anything from first prediction value
            previous_timesteps[1:] = pred[:-1]

            if previous_pred is not None:
                pred[0] = pred[0] - previous_pred
            else:
                # pred[0] = 0 - pred[0]
                pred[0] = pred[0]
            pred -= previous_timesteps
            # enforce same return
            returns = torch.sum(torch.tensor(episode.rewards[-to_add:], device=self.device))
            predicted_returns = pred.sum()
            prediction_error = returns - predicted_returns
            pred += prediction_error / pred.shape[0]

            self.replay_buffer.current_predictions[proc_id].append(pred.detach().clone())

    def new_predict_reward(self, dones, timestep):
        # self.net.eval()
        for i, done in enumerate(dones):
            if done and timestep != self.frames_per_proc - 1:
                self.do_part_prediction(i, timestep)
            if timestep == self.frames_per_proc - 1:
                self.do_part_prediction(i, timestep)

        if timestep == self.frames_per_proc - 1:
            # output shape = (self.num_frames_per_proc, self.num_procs)
            all_frames = []
            for proc_id, _ in enumerate(dones):
                all_frames.append(torch.cat(self.replay_buffer.current_predictions[proc_id], dim=-1))
                self.replay_buffer.current_predictions[proc_id] = []
            # the first time we have incomplete episodes as an input
            # which started before training was done so stack will fail if not all records are complete
            try:
                ret = torch.stack(all_frames)
            except:
                ret = None
            return ret
