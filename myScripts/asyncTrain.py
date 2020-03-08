import time

def my_callback(argi):
    print("reached callback")
    print(argi)
def start_background_process(rudder,queue_in,queue_out):
# def start_background_process(rudder,replay_bufffer):

    print("in proc")
    while True:
        time.sleep(0.1)
        if not queue_in.empty():

            print("async train start")
            replay_bufffer=queue_in.get()
            rudder.replay_buffer=replay_bufffer
            rudder.train_full_buffer()
            queue_out.put(True)
            print("async out")
            with open(rudder.communication_file_path,"w") as file:
                file.write("done")



