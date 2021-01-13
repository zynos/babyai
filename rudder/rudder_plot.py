import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


class RudderPlotter:
    def __init__(self,il_learn):
        self.total_pics = 0
        self.out_image_height = 10.8
        self.out_image_width = 19.2
        self.il_learn = il_learn

        Path("myPics").mkdir(parents=True, exist_ok=True)

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

    def plot_current_step(self, orig_rews, predictions, actions, ax, i, label, plot_orig):
        action_dict = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up", 4: "drop", 5: "toggle",
                       6: "done"}
        actions = [action_dict[a.item()] for a in actions]
        redistributed_rews = self.redistribute_reward(predictions.unsqueeze(0),orig_rews.unsqueeze(0)).cpu().squeeze().numpy()
        # predictions = predictions.cpu().squeeze().numpy()
        if plot_orig:
            rews = orig_rews.cpu().clone().numpy()
            # rews = self.il_learn.scale_rewards(rews, self.il_learn.minus_to_one_scale)
            ax.plot(rews, label="original rewards")
        ax.plot(redistributed_rews, label="redistributed rewards " + str(label))
        ax.set_xticks(list(range(len(actions))))
        ax.set_xticklabels(actions, rotation=90)
        ax.legend(loc="upper left")
        ax.stem([i], [redistributed_rews[i]], linefmt="r--", markerfmt="r")
        ax.get_xticklabels()[i].set_color("red")
        return actions[i]

    def plot_reward_redistribution(self, start, stop, path_start, model_predictions, short_episode, env,
                                   top_titel=None):
        # loss, returns, quality, predictions = self.rudder.feed_network(short_episode)
        if not isinstance(short_episode.instructions[0],str):
            command = {"put": 1, "the": 2, "grey": 3, "key": 4, "next": 5, "to": 6, "red": 7, "box": 8, "yellow": 9,
                       "blue": 10, "green": 11, "purple": 12, "ball": 13}
            inv_map = {v: k for k, v in command.items()}
            lis = [inv_map[i.item()] for i in short_episode.instructions[0].cpu()]
            command = " ".join(lis)
        else:
            command = short_episode.instructions[0]
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
            if top_titel:
                print("expect", top_titel)
            for el in model_predictions:
                print("loss, main, aux", el[2], el[1])
                action = self.plot_current_step(short_episode.rewards, el[0], short_episode.actions,
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
