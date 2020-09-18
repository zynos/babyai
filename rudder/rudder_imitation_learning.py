import json
import logging
import os
import time
import datetime
import pickle
from collections import Counter

import gym
import numpy as np
import torch
from babyai import utils
from rudder.rudder_model import ACModel

logger = logging.getLogger(__name__)


# copy from imitation.py
class EpochIndexSampler:
    """
    Generate smart indices for epochs that are smaller than the dataset size.

    The usecase: you have a code that has a strongly baken in notion of an epoch,
    e.g. you can only validate in the end of the epoch. That ties a lot of
    aspects of training to the size of the dataset. You may want to validate
    more often than once per a complete pass over the dataset.

    This class helps you by generating a sequence of smaller epochs that
    use different subsets of the dataset, as long as this is possible.
    This allows you to keep the small advantage that sampling without replacement
    provides, but also enjoy smaller epochs.
    """

    def __init__(self, n_examples, epoch_n_examples):
        self.n_examples = n_examples
        self.epoch_n_examples = epoch_n_examples

        self._last_seed = None

    def _reseed_indices_if_needed(self, seed):
        if seed == self._last_seed:
            return

        rng = np.random.RandomState(seed)
        self._indices = list(range(self.n_examples))
        rng.shuffle(self._indices)
        logger.info('reshuffle the dataset')

        self._last_seed = seed

    def get_epoch_indices(self, epoch):
        """Return indices corresponding to a particular epoch.

        Tip: if you call this function with consecutive epoch numbers,
        you will avoid expensive reshuffling of the index list.

        """
        seed = epoch * self.epoch_n_examples // self.n_examples
        offset = epoch * self.epoch_n_examples % self.n_examples

        indices = []
        while len(indices) < self.epoch_n_examples:
            self._reseed_indices_if_needed(seed)
            n_lacking = self.epoch_n_examples - len(indices)
            indices += self._indices[offset:offset + min(n_lacking, self.n_examples - offset)]
            offset = 0
            seed += 1

        return indices


class RudderImitation(object):
    def __init__(self, path_to_demos,use_actions, args):
        self.max_len = 128
        self.minus_to_one_scale = True
        self.use_actions = use_actions
        self.use_rudder = True
        self.epochs = 10
        self.args = args
        self.aux_loss_multiplier = 0.1
        self.env = gym.make(self.args.env)
        self.calc_and_set_mean_and_stddev_from_episode_lens(path_to_demos)
        # demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
        # demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)
        #
        # logger.info('loading demos')
        # self.train_demos = utils.load_demos(demos_path)
        # logger.info('loaded demos')
        # if args.episodes:
        #     if args.episodes > len(self.train_demos):
        #         raise ValueError("there are only {} train demos".format(len(self.train_demos)))
        #     self.train_demos = self.train_demos[:args.episodes]
        #
        # self.val_demos = utils.load_demos(demos_path_valid)
        # if args.val_episodes > len(self.val_demos):
        #     logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
        # self.val_demos = self.val_demos[:self.args.val_episodes]

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None))

        # Define actor-critic model
        self.acmodel = utils.load_model(args.model, raise_not_found=False)
        if self.acmodel is None:
            if getattr(self.args, 'pretrained_model', None):
                self.acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
            else:
                logger.info('Creating new model')
                self.acmodel = ACModel(self.obss_preprocessor.obs_space, action_space,
                                       args.image_dim, args.memory_dim, args.instr_dim,
                                       not self.args.no_instr, self.args.instr_arch,
                                       not self.args.no_mem, self.args.arch,without_action=not use_actions)
        self.obss_preprocessor.vocab.save()
        utils.save_model(self.acmodel, args.model)

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def default_model_name(action_input,args):
        if getattr(args, 'multi_env', None):
            # It's better to specify one's own model name for this scenario
            named_envs = '-'.join(args.multi_env)
        else:
            named_envs = args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'action': action_input,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{arch}_{instr}_seed{seed}_actionIn_{action}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def calculate_rewards_from_length(self, lens, max_len):
        rewards = [1 - 0.9 * (l / max_len) if l != max_len else 0 for l in lens]
        return rewards

    def calc_and_set_mean_and_stddev_from_episode_lens(self, path):
        if not "train" in path:
            path += "train/"
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
        rewards = self.calculate_rewards_from_length(lens, self.max_len)
        self.mean = np.mean(rewards)
        self.std_dev = np.std(rewards)

    def scale_rewards_minus_one_to_1(self, rewards):
        rewards = (rewards - self.mean) / self.std_dev
        return rewards

    def scale_rewards(self, rewards, minus_to_one_scale=False):
        if minus_to_one_scale:
            rewards = self.scale_rewards_minus_one_to_1(rewards)
        else:
            rewards *= self.reward_scale
        return rewards

    def calculate_reward(self, step_count, max_steps):
        if step_count == max_steps:
            return 0.0
        return (1 - 0.9 * (step_count / max_steps)) * 1

    def my_stuff(self, batch):
        lens = [len(episode) for episode in batch]
        # only 0 required because its sorted
        max_steps = 128
        assert lens[0] <= max_steps
        rewards = [self.calculate_reward(l, max_steps) for l in lens]
        if self.minus_to_one_scale:
            rewards = self.scale_rewards(rewards, self.minus_to_one_scale)
            empty_rewards = [self.scale_rewards(torch.zeros(l, device=self.device), self.minus_to_one_scale) for l in
                             lens]
            # empty_rewards = torch.from_numpy(empty_rewards).to(self.device)
        for i, reward in enumerate(rewards):
            empty_rewards[i][-1] = reward
        repeated_rewards = []
        for i, len_ in enumerate(lens):
            repeated_rewards.append(torch.tensor(rewards[i], device=self.device).expand(len_))

        return empty_rewards, repeated_rewards

    def calculate_my_loss(self, predicted_reward, reward_empty_step, my_done_step, reward_repeated_step):
        main_loss = (((predicted_reward - reward_empty_step) * my_done_step) ** 2).mean()
        aux_loss = ((predicted_reward - reward_repeated_step) ** 2).mean()
        final_loss = main_loss + self.aux_loss_multiplier * aux_loss
        return final_loss, (main_loss.detach().clone(), aux_loss.detach().clone())

    def filter_finished_episodes(self, list_of_tuples):
        out = []
        for i, tup in enumerate(list_of_tuples):
            dones = tup[3]
            if dones.any():
                to_finish = np.argwhere(dones == True)
                for index in to_finish:
                    predictions = [e[0][index] for e in list_of_tuples[:i + 1]]
                    orig_rewards = [e[1][index] for e in list_of_tuples[:i + 1]]
                    dones = [e[3][index] for e in list_of_tuples[:i + 1]]
                    actions = [e[4][index] for e in list_of_tuples[:i + 1]]
                    obs = [e[5][index] for e in list_of_tuples[:i + 1]]
                    out.append((predictions, orig_rewards, dones, actions, obs, self.args.model))
        return out

    def run_epoch_recurrence(self, demos, is_training=False, indices=None, rudder_eval=False):
        if not indices:
            indices = list(range(len(demos)))
            if is_training:
                np.random.shuffle(indices)
        batch_size = min(self.args.batch_size, len(demos))
        offset = 0

        if not is_training:
            self.acmodel.eval()

        # Log dictionary
        log = {"entropy": [], "policy_loss": [], "accuracy": [], "aux_loss": [], "main_loss": []}

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            logger.info("batch {}, FPS so far {}".format(
                batch_index, frames / (time.time() - start_time) if frames else 0))
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += sum([len(demo[3]) for demo in batch])

            if rudder_eval:
                _log, finished_episodes = self.run_epoch_recurrence_one_batch(batch, is_training=is_training,
                                                                              rudder_eval=rudder_eval)
            else:
                _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training, rudder_eval=rudder_eval)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])
            log["aux_loss"].append(_log["aux_loss"])
            log["main_loss"].append((_log["main_loss"]))

            offset += batch_size
        log['total_frames'] = frames

        if not is_training:
            self.acmodel.train()

        if rudder_eval:
            return log, finished_episodes
        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, rudder_eval=False):
        batch = utils.demos.transform_demos(batch)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]
        if self.use_rudder:
            empty_rewards, repeated_rewards = self.my_stuff(batch)

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)
        if self.use_rudder:
            reward_empty_true = torch.cat(empty_rewards)
            reward_repeated_true = torch.cat(repeated_rewards)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device)
        instr_embedding = self.acmodel._get_instr_embedding(preprocessed_first_obs.instr)

        # Loop terminates when every observation in the flat_batch has been handled
        my_rews = []
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            action_step2 = action_true[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                model_res = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)],action_step2)
                new_memory = model_res['memory']
                pred_rew = model_res["value"]
                my_rews.append(
                    (pred_rew, reward_empty_true[inds], reward_repeated_true[inds], done_step, action_true[inds], obs))

            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]
        if rudder_eval:
            finished_episodes = self.filter_finished_episodes(my_rews)

        # Here, actual backprop upto args.recurrence happens
        final_loss, final_main_loss, final_aux_loss = 0, 0, 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0
        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        rewards = []
        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            if self.use_rudder:
                reward_empty_step = reward_empty_true[indexes]
                reward_repeated_step = reward_repeated_true[indexes]
                my_done_step = torch.from_numpy(done[indexes].astype(bool)).float().to(self.device)

            action_step = action_true[indexes]
            mask_step = mask[indexes]
            model_results = self.acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]],action_step)
            if self.use_rudder:
                predicted_reward = model_results['value']
                rewards.append((predicted_reward, reward_empty_step, reward_repeated_step, my_done_step))
            dist = model_results['dist']
            memory = model_results['memory']

            entropy = dist.entropy().mean()
            if self.use_rudder:
                policy_loss, (main_loss, aux_loss) = self.calculate_my_loss(predicted_reward, reward_empty_step,
                                                                            my_done_step,
                                                                            reward_repeated_step)
                final_main_loss += main_loss
                final_aux_loss += aux_loss
                loss = policy_loss
                accuracy = 0.0
            else:
                policy_loss = -dist.log_prob(action_step).mean()
                loss = policy_loss - self.args.entropy_coef * entropy
                final_main_loss += 0
                final_aux_loss += 0
                action_pred = dist.probs.max(1, keepdim=True)[1]
                accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss.detach().clone()
            indexes += 1

        final_loss /= self.args.recurrence
        final_aux_loss /= self.args.recurrence
        final_main_loss /= self.args.recurrence

        log = {}
        grad_norm = 0.0

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            grad_norm = sum(
                p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
            self.optimizer.step()
            # Learning rate scheduler
        log["grad_norm"] = float(grad_norm)
        print("loss, grad norm", final_loss.item(), log["grad_norm"])
        log["entropy"] = float(final_entropy / self.args.recurrence)
        log["policy_loss"] = float(final_policy_loss / self.args.recurrence)
        log["accuracy"] = float(accuracy)
        log["aux_loss"] = float(final_aux_loss)
        log["main_loss"] = float(final_main_loss)
        if rudder_eval:
            return log, finished_episodes
        return log

    def get_train_and_validate_demo_files(self, path_to_demos):
        train_files = os.listdir(path_to_demos + "train/")
        valid_files = os.listdir(path_to_demos + "validate/")
        train_files.sort()
        valid_files.sort()
        return train_files, valid_files

    def load_demos(self, path):
        with open(path, "rb") as f:
            res = pickle.load(f)
        return res

    def train(self, path_to_demos, writer, csv_writer, status_path, header, reset_status=False):
        # Load the status
        def initial_status():
            return {'i': 0,
                    'num_frames': 0,
                    'patience': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))
        train_files, valid_files = self.get_train_and_validate_demo_files(path_to_demos)

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.acmodel, self.args.model)

        # best mean return to keep track of performance on validation set
        best_loss, patience, i = 1000, 0, 0
        total_start_time = time.time()

        for _ in range(self.epochs):
            for train_f, valid_f in zip(train_files, valid_files):
                train_demos = self.load_demos(path_to_demos + "train/" + train_f)
                val_demos = self.load_demos(path_to_demos + "validate/" + valid_f)

                # If the batch size is larger than the number of demos, we need to lower the batch size
                if self.args.batch_size > len(train_demos):
                    self.args.batch_size = len(train_demos)
                    logger.info(
                        "Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

                epoch_length = self.args.epoch_length
                if not epoch_length:
                    epoch_length = len(train_demos)
                index_sampler = EpochIndexSampler(len(train_demos), epoch_length)

                self.train_one_file(status, index_sampler, best_loss, total_start_time, train_demos, val_demos,
                                    writer,
                                    csv_writer, status_path, header, reset_status)

            status['i'] += 1

    def train_one_file(self, status, index_sampler, best_loss, total_start_time, train_demos, val_demos, writer,
                       csv_writer, status_path, header, reset_status=False):

        # while status['i'] < getattr(self.args, 'epochs', int(1e9)):
        if 'patience' not in status:  # if for some reason you're finetuining with IL an RL pretrained agent
            status['patience'] = 0
        # Do not learn if using a pre-trained model that already lost patience
        if status['patience'] > self.args.patience:
            return
        if status['num_frames'] > self.args.frames:
            return

        update_start_time = time.time()

        indices = index_sampler.get_epoch_indices(status['i'])
        assert len(indices) == len(train_demos)
        log = self.run_epoch_recurrence(train_demos, is_training=True, indices=indices)

        # Learning rate scheduler
        self.scheduler.step()

        status['num_frames'] += log['total_frames']

        update_end_time = time.time()

        # Print logs
        if status['i'] % self.args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)

            fps = log['total_frames'] / (update_end_time - update_start_time)
            # duration = datetime.timedelta(seconds=total_ellapsed_time)

            for key in log:
                log[key] = np.mean(log[key])

            train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                          log["entropy"], log["policy_loss"], log["accuracy"], log["main_loss"],
                          log["aux_loss"]]

            logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f},  mainL {: .5f}  auxL {: .3f}".format(
                    *train_data))

            # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
            # afterwards when status['i'] % self.args.val_interval == 0
            if status['i'] % self.args.val_interval != 0:
                # instantiate a validation_log with empty strings when no validation is done
                validation_data = [''] * len([key for key in header if 'valid' in key])
                assert len(header) == len(train_data + validation_data)
                if self.args.tb:
                    for key, value in zip(header, train_data):
                        writer.add_scalar(key, float(value), status['num_frames'])
                csv_writer.writerow(train_data + validation_data)

        if status['i'] % self.args.val_interval == 0:

            # valid_log = self.validate(self.args.val_episodes)
            # mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
            # success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
            #                 valid_log]

            val_log = self.run_epoch_recurrence(val_demos)
            loss = np.mean(val_log["policy_loss"])
            loss_main = np.mean(val_log["main_loss"])
            loss_aux = np.mean(val_log["aux_loss"])

            if status['i'] % self.args.log_interval == 0:
                # validation_data = [validation_accuracy] + mean_return + success_rate
                validation_data = [loss,loss_main,loss_aux]

                assert len(header) == len(train_data + validation_data)
                if self.args.tb:
                    for key, value in zip(header, train_data + validation_data):
                        writer.add_scalar(key, float(value), status['num_frames'])
                csv_writer.writerow(train_data + validation_data)

            # In case of a multi-env, the update condition would be "better mean success rate" !
            if np.mean(loss) < best_loss:
                best_loss = np.mean(loss)
                status['patience'] = 0
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)
                # Saving the model
                logger.info("Saving best model")

                if torch.cuda.is_available():
                    self.acmodel.cpu()
                utils.save_model(self.acmodel, self.args.model + "_best")
                self.obss_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                if torch.cuda.is_available():
                    self.acmodel.cuda()
            else:
                status['patience'] += 1
                logger.info(
                    "Losing patience, new value={}, limit={}".format(status['patience'], self.args.patience))

            if torch.cuda.is_available():
                self.acmodel.cpu()
            utils.save_model(self.acmodel, self.args.model)
            self.obss_preprocessor.vocab.save()
            if torch.cuda.is_available():
                self.acmodel.cuda()
            with open(status_path, 'w') as dst:
                json.dump(status, dst)

        return best_loss
