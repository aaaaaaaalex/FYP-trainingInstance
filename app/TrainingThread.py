import threading
from trainingInstance import TrainingInstance

class TrainingThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name)
        self.args = args
        self.kwargs = kwargs
        return


    def run(self):
        instance = TrainingInstance(
            config_path=self.args.config,
            checkpoint_path=self.args.checkpoint_dir,
            use_dataset_cache=self.args.skip_download,
            min_images_per_class=self.args.min_images_per_class)
        
        # start training
        instance.init()
        instance.train()
        instance.save_model()

        return
