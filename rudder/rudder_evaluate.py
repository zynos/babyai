import logging
import os

import numpy as np
import torch
from babyai import utils
from babyai.arguments import ArgumentParser
from rudder.rudder_imitation_learning import RudderImitation
from myScripts.ReplayBuffer import ProcessData
from rudder.rudder_plot import RudderPlotter
import matplotlib.image as mpimg

parser = ArgumentParser()
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--multi-env", nargs='*', default=None,
                    help="name of the environments used for validation/model loading")
parser.add_argument("--multi-demos", nargs='*', default=None,
                    help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
parser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")
parser.add_argument("--epoch-length", type=int,
                    help="number of examples per epoch")


def build_loss_str(e):
    return "loss {:.4f} main {:.4f} aux {:.4f} lastPred {:.3} {} ".format(float(e[0]), float(e[1]),
                                                                          float(e[2]), float(e[3][-1]),
                                                                          e[4])


def evaluate(finished_episode, il_learn, i):
    episode = ProcessData()
    first_ep = finished_episode
    predictions, orig_rewards, dones, actions, obs, model_name = first_ep
    assert obs[0][0]["mission"] == obs[-1][0]["mission"]
    # calculate loss
    reward_repeated_step = orig_rewards[-1].expand(len(dones))
    assert len(reward_repeated_step) == len(dones)
    predictions = torch.tensor(predictions, device=il_learn.device)
    orig_rewards = torch.tensor(orig_rewards, device=il_learn.device)
    episode.rewards = orig_rewards
    episode.instructions = [obs[0][0]["mission"]]
    episode.images = torch.tensor([e[0]["image"] for e in obs], device=il_learn.device)
    episode.actions = actions
    my_done_step = torch.from_numpy(np.array(dones).astype(bool)).float().to(il_learn.device).flatten()
    final_loss, (main_loss, aux_loss) = il_learn.calculate_my_loss(predictions, orig_rewards, my_done_step,
                                                                   reward_repeated_step)
    loss_str = build_loss_str((final_loss, main_loss, aux_loss, predictions, model_name))
    rudder_plotter = RudderPlotter(il_learn)
    # model pred contains (predictions.squeeze(), model_file_name[:-3], (loss, main_loss, aux_loss))
    model_predictions = [(predictions, model_name, (final_loss, (main_loss, aux_loss)))]
    rudder_plotter.plot_reward_redistribution("0", str(len(episode.images)) + "_" + str(i), "myOutput5/",
                                              model_predictions, episode,
                                              il_learn.env, top_titel=loss_str)


def main(path_to_demos, args):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)

    if not args.model:
        print("specify model with --model")
        return

    args.model = args.model or RudderImitation.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    il_learn = RudderImitation(path_to_demos, args)
    valid_files = os.listdir(path_to_demos + "validate/")
    valid_demos = il_learn.load_demos(path_to_demos + "validate/" + valid_files[0])
    log, finished_episodes = il_learn.run_epoch_recurrence(valid_demos, rudder_eval=True)

    for i in range(15):
        evaluate(finished_episodes[i], il_learn, i)
        evaluate(finished_episodes[-i], il_learn, -i)


if __name__ == "__main__":
    args = parser.parse_args()
    main("../scripts/demos/240kDS/", args)
