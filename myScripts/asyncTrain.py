

def start_background_process(rudder,queue_in,queue_out):
    print("in proc")
    while True:
        if not queue_in.empty():
            args=queue_in.get()
            rudder.add_timestep_data(*args)
            if rudder.first_training_done:
                queue_out.put(rudder.first_training_done)



