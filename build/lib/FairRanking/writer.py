from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class Writer():

    def __init__(self,
                 hidden_layers: list,
                 schedule: list,
                 scheduler_run: str,
                 dataset: str,
                 experiment_name: str,
                 model_name: str,
                 extra: str = None): 
                   
        timestamp = datetime.now().strftime('%Y-%m-%d')

        if extra:
            log_dir = os.path.join("runs", dataset, timestamp, scheduler_run, experiment_name, model_name, extra)
        else:
            log_dir = os.path.join("runs", dataset, timestamp, experiment_name, model_name, extra)
        print(f"[INFO] Created SummaryWriter saving to {log_dir}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text(tag='Hidden Layers', text_string=str(hidden_layers))
        self.writer.add_text(tag='Schedule', text_string=str(schedule))
    
    
    def write(self, **kwargs):
        for key, values in kwargs.items():
            self.writer.add_scalars(main_tag=key,
                                    tag_scalar_dict=values[0],
                                    global_step = values[1])
            
    
    def __del__(self):
        self.writer.close()
    

