import threading
from TrainingInstance import TrainingInstance

class TrainingThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name)
        self.args = args
        self.kwargs = kwargs
        return


    def run(self):
        instance = TrainingInstance(
            config_path=self.args['config'] if 'config' in self.args else None,
            checkpoint_path=self.args['checkpoint_dir'] if 'checkpoint_dir' in self.args else None,
            use_dataset_cache=self.args['skip_download'] if 'skip_download' in self.args else None,
            min_images_per_class=self.args['min_images_per_class'] if 'min_images_per_class' in self.args else None)
        
        # start training
        instance.init()
        instance.train()
        instance.save_model()

        return
