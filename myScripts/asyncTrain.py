
import time
def start_background_process(rudder,queue_in,queue_out):
    print("in proc")
    while True:
        time.sleep(0.1)
        if not queue_in.empty():
            replay_buffer=queue_in.get()
            rudder.replay_buffer=replay_buffer
            rudder.train_full_buffer()
            queue_out.put(True)



