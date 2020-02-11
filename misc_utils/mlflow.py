from keras.callbacks import Callback
import mlflow


class MlflowCallback(Callback):
    """Track experiment using Mlflow
    """
    
    def __init__(self, experiment_id: int, run_name: str):
        """All metrics will be track after completion of everu
        
        Parameters
        ----------
        experiment_id : int
            Mlflow experiment
        run_name : str
            Name to be given to current run
        """
        self.experiment_id = experiment_id
        self.run_name = run_name
        
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric('loss',logs['loss'])
        mlflow.log_metric('val_loss',logs['val_loss'])
        print(f'At end of Epoch {epoch} loss is {logs['loss']:.4f} and val_loss is {logs['val_loss']:.4f}')
    
    