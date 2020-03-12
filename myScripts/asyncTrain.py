import multiprocessing as mp
import logging
import time

def start_background_process(rudder,queue_in,queue_out):
    mpl = mp.log_to_stderr()
    mpl.setLevel(logging.INFO)
    print("in proc")
    while True:
        time.sleep(0.1)
        if not queue_in.empty():
            # print("queue in",queue_in)
            # print("queue out", queue_out)
            replay_buffer=queue_in.get()
            rudder.replay_buffer=replay_buffer
            rudder.train_full_buffer()
            # print("wanna feed queue")
            # print("queue in", queue_in)
            # print("queue out", queue_out)
            # print(queue_out.empty())
            queue_out.put(True)
            print("after background proc put")



