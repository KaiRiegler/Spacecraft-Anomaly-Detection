import gin
import tensorflow as tf
import logging
import wandb
import numpy as np
import matplotlib.pyplot as plt

@gin.configurable
class Trainer(object):
    '''
    A class for training and validating a deep learning model with logging and checkpointing.

        Attributes:
            model (tf.keras.Model): The neural network model.
            ds_train (tf.data.Dataset): Training dataset.
            ds_val (tf.data.Dataset): Validation dataset.
            ds_info (dict): Information about the dataset.
            run_paths (dict): Paths for logs, checkpoints, and configuration.
            lr (float): Initial learning rate.
            total_steps (int): Total number of training steps.
            log_interval (int): Steps between logging training progress.
            ckpt_interval (int): Steps between saving checkpoints.
            run (wandb.Run): Weights & Biases run instance.
    '''
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, lr, total_steps, log_interval, ckpt_interval, run):

        # Loss objective
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval 
        self.run = run     
        
        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, step=tf.Variable(1))
        self.manager = tf.train.CheckpointManager(self.ckpt, run_paths['path_ckpts_train'], max_to_keep=1)

    @gin.configurable
    def plot_result(self, ds, ls, win_size, win_offset):
        '''
        Plot the actual signal and the reconstructed signal.

            Parameters:
                self (self): Trainer
                ds (Dataset): Dataset with the actual signal
                ls (int): Length of signal (before windowing)
                win_size (int): Length of the window
                win_offset (int): Shift of the sliding window  

            Returns:
                None
        '''
        d_true = np.array([])
        d_pred = np.array([])
        l = ls - (ls % win_offset) # length of signal minus lost samples
        self.val_loss.reset_states()
        for batch in ds:
            predictions = self.val_step(batch)
            for w_true, w_pred in zip(batch, predictions):
                if win_size == win_offset: 
                    d_true = np.append(d_true, w_true.numpy())
                    d_pred = np.append(d_pred, w_pred.numpy().flatten())
                else:
                    d_true = np.append(d_true, w_true.numpy()[:win_offset])
                    d_pred = np.append(d_pred, w_pred.numpy().flatten()[:win_offset])

                if len(d_true) >= l: # stop after window count for one epoch                   
                    if l != len(d_true):
                        d_true = d_true[:l]
                        d_pred = d_pred[:l]
                    x = np.linspace(0, l-1, l) 
                    plt.figure(0, figsize=(15, 6))
                    plt.plot(x, d_true, color='green', label="True signal")
                    plt.plot(x, d_pred, color='red', label="Reconstructed signal")
                    plt.fill_between(x, d_true, d_pred)
                    plt.title("Train: True vs. Reconstruction")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Signal Value")
                    plt.legend()
                    plt.grid(True)
                    wandb.log({"Train result": wandb.Image(plt)})
                    plt.close()
                    return None

    @tf.function
    def train_step(self, batch):
        '''
        Performs a single training step.

        Parameters:
            batch (tf.Tensor): Batch of training sequences.

        Returns:
            tf.Tensor: Reconstructed sequence.
        '''
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(batch, training=True)
            loss = self.loss_object(batch, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        return predictions

    @tf.function
    def val_step(self, batch):
        '''
        Performs a single validation step.

        Parameters:
            batch (tf.Tensor): Batch of validation sequences.

        Returns:
            tf.Tensor: Reconstructed sequence.
        '''
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(batch, training=False)
        t_loss = self.loss_object(batch, predictions)

        self.val_loss.update_state(t_loss)
        return predictions

    def train(self):
        '''
        Executes the training loop with logging and checkpointing.

        Yields:
            float: Validation accuracy at each logging interval.
        '''
        # restoring latest checkpoint
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            logging.info(f'Restored checkpoint from {self.manager.latest_checkpoint}.')

        for idx, batch in enumerate(self.ds_train):
            self.ckpt.step.assign_add(1)
            step = int(self.ckpt.step)
            self.train_step(batch)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_loss.reset_states()
                
                for val_batch in self.ds_val:
                    self.val_step(val_batch)

                template = 'Train: Step {}, Loss: {}'
                logging.info(template.format(step,
                                             self.train_loss.result()))
                template = 'Validation: Step {}, Loss: {}'
                logging.info(template.format(step,
                                             self.val_loss.result()))
                
                # Write summary to wandb
                wandb.log({'train_loss': self.train_loss.result(),
                           'val_loss': self.val_loss.result()
                           })

                # Reset train metrics
                self.train_loss.reset_states()

                yield self.val_loss.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                artifact = wandb.Artifact(name="model-"+self.run.name, type='model')
                artifact.add_dir(local_path=self.run_paths["path_model_id"])
                self.run.log_artifact(artifact)
                return self.val_loss.result().numpy()
