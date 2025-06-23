import os
from veesion_data_core.tools import (
    get_training_request,
    request_training,
    resume_training,
    stop_training,
)
from datetime import datetime, timezone

date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

training_name = f"llava_cheater-28" # Focus debug on timestamp generation
simone_branch = "local_debug_hang"

request_training(
    training_name=training_name,
    n_epochs=50,
    dataset_version=28,
    cpu_instances_count=0,
    gpu_instances_count=1,
    terraform_scalable_training_branch="llava_training",
    simone_branch=simone_branch,
)

print(f"Training request '{training_name}' submitted.")

# # # Get the training request

# training_request = get_training_request(training_name=training_name)


# print("Training name:", training_request.training_name)
# print("Training status:", training_request.status)
# print("Number of Training Run attempted ", training_request.scheduling_attempt)
# # stop_training(training_name=training_name)
# import time
# # Removed the following while loop
# # while 1:
# #     try:
# #         resume_training(training_name=training_name, force_resume=True)
# #     except:
# #         pass
# #     time.sleep(600)
