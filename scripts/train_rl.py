#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""
from apex import amp
import os
import logging
import csv
import json
from collections import Counter
from torch.multiprocessing.queue import Queue
import gym
import time
import datetime
import torch

# torch.backends.cudnn.benchmark = True
import numpy as np
import subprocess
import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from torch import multiprocessing as mp
print("cuda",torch.version.cuda)
print("cudnn",torch.backends.cudnn.version())

def train_old_samples(rudder,queue):
# def train_old_samples(rudder):
    end = False
    ids = []
    new_sample_losses=[]
    while not end:
        qualitys = []
        # print("queue size",queue.qsize())
        for i in range(rudder.rudder_train_samples_per_epochs):
            sample = rudder.replayBuffer.get_sample()
            loss, quality =  rudder.train_rudder(sample)
            new_sample_losses.append((loss.detach().clone().item() , sample["id"]))
            # queue.put((loss.detach() , sample["id"]))
            # rudder.replayBuffer.update_sample_loss(loss, sample["id"])
            ids.append(sample["id"])
            # print("loss {}, quality {}, sample {} ".format(loss.item(), quality.item(), sample["id"]))
            qualitys.append((quality >= 0).item())
        print("loss, qualities",loss.item(), qualitys)

        if False in qualitys:
            end = False
        else:
            end = True
    idc = Counter(ids)
    rudder.current_loss = loss
    torch.cuda.empty_cache()
    rudder.replayBuffer.added_new_sample = False
    rudder.training_done = True
    # self.rudder_net.lstm1.plot_internals(filename=None, show_plot=True, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))
    # ret=copy.deepcopy(rudder.replayBuffer)
    queue.put(new_sample_losses, block=True)
    # print("before queue put")
    queue.put("end", block=True)

    # queue.put(new_sample_losses)
    # print(queue)
    # queue.join()
    # print("exitiging")
    queue.close()
    return new_sample_losses

def my_callback( argi):
    print(argi)
    print(
        "ended the process   ##################################################################################################################################")


def my_err_callback( lol):
    print(lol)
    print(
        "failed the process   ##################################################################################################################################")

if __name__ == '__main__':
    # torch.set_num_threads(1)
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use (default: ppo)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--reward-scale", type=float, default=20.,
                        help="Reward scale multiplier")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    args = parser.parse_args()

    utils.seed(args.seed)

    use_bert = False
    bert_dim = 512
    use_rudder= True
    rudder_own_net = True
    use_reshaped_reward = True
    # Generate environments
    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        env.seed(100 * args.seed + i)
        envs.append(env)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'info': '',
        'coef': '',
        'suffix': suffix}
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
    args.model = args.model.format(**model_name_parts) if args.model else default_model_name

    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    # Define obss preprocessor
    if 'emb' in args.arch:
        # obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
        obss_preprocessor = None
    else:
        obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model,
                                                   use_bert=use_bert,bert_dim=bert_dim)

    # Define actor-critic model
    acmodel = utils.load_model(args.model, raise_not_found=False)
    if acmodel is None:
        if args.pretrained_model:
            acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
        else:
            instr_dim=bert_dim if use_bert else 128
            acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                              args.image_dim, args.memory_dim, instr_dim,
                              not args.no_instr, args.instr_arch, not args.no_mem, args.arch,use_bert=use_bert)

    obss_preprocessor.vocab.save()
    utils.save_model(acmodel, args.model)

    if torch.cuda.is_available():
        acmodel.cuda()

    # Define actor-critic algo

    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    if not use_reshaped_reward:
        reshape_reward = None
    if args.algo == "ppo":
        algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                                 args.gae_lambda,
                                 args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                 args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                 reshape_reward,use_rudder=use_rudder,rudder_own_net=rudder_own_net,env_max_steps=env.max_steps)

        ctx = mp.get_context('spawn')
        # pool = ctx.Pool(1, maxtasksperchild=1)
        # algo.parallel_train_func=train_old_samples
        # algo.ctx = mp.get_context('spawn')
        # self.p = ctx.Process(target=self.parallel_train_func, args=(self.rudder,))
        algo.parallel_train_func=train_old_samples
        # algo.pool=pool
        algo.my_callback=my_callback
        algo.my_error_callback=my_err_callback
        algo.ctx=ctx
        algo.pool = algo.ctx.Pool(1, maxtasksperchild=1)
        algo.queue=ctx.Queue()
        algo.p = ctx.Process(target= algo.parallel_train_func, args=(algo.rudder,algo.queue,))
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status

    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and Tensorboard writer and CSV writer

    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"]
              + ["RUD_L"])
    if args.tb:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(utils.get_log_dir(args.model))
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Log code state, command, availability of CUDA and model

    babyai_code = list(babyai.__path__)[0]
    try:
        last_commit = subprocess.check_output(
            'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
        logger.info('LAST COMMIT INFO:')
        logger.info(last_commit)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    try:
        diff = subprocess.check_output(
            'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
        if diff:
            logger.info('GIT DIFF:')
            logger.info(diff)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(acmodel)

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    test_env_name = args.env
    while status['num_frames'] < args.frames:
        # Update parameters

        update_start_time = time.time()
        print("start algo.update_parameters()")
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        if 'num_episodes' not in status.keys():
            status['num_episodes']=0
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs

        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"],logs["RUD_L"]]

            format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                          "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                          "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | RUD_L {:.3f}")

            logger.info(format_str.format(*data))
            if args.tb:
                assert len(header) == len(data)
                for key, value in zip(header, data):
                    writer.add_scalar(key, float(value), status['num_frames'])

            csv_writer.writerow(data)

        # Save obss preprocessor vocabulary and model

        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            obss_preprocessor.vocab.save()
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_model(acmodel, args.model)

            # Testing the model before saving
            agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
            agent.model = acmodel
            agent.model.eval()
            logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes)
            agent.model.train()
            mean_return = np.mean(logs["return_per_episode"])
            success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                utils.save_model(acmodel, args.model + '_best')
                obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
                logger.info("Return {: .2f}; best model is saved".format(mean_return))
            else:
                logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
