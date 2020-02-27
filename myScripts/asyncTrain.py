

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
            ids.append(sample["id"])
            del loss
            del sample
            # torch.cuda.empty_cache()
            # queue.put((loss.detach() , sample["id"]))
            # rudder.replayBuffer.update_sample_loss(loss, sample["id"])
            # print("loss {}, quality {}, sample {} ".format(loss.item(), quality.item(), sample["id"]))
            qualitys.append((quality >= 0).item())
        print("loss, qualities",new_sample_losses[-1][0], qualitys)

        if False in qualitys:
            end = False
        else:
            end = True
    # idc = Counter(ids)
    # torch.cuda.empty_cache()
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