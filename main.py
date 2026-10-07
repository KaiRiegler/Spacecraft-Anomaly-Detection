import os
import gin
import tensorflow as tf
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import LSTM_Autoencoder


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('run', '', 'Specify run id (path to folder in experiments). If folder exists continue training from last checkpoint.')
flags.DEFINE_string('wandb', '', 'Specify how wandb syncs runs. default: no sync, <config>: sync with api key from config, <your-api-key>: sync with your-api-key')


def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.run)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings([os.path.join(os.path.dirname(__file__), 'configs/config.gin')], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # initialize wandb
    run = utils_params.wandb_init(run_paths, FLAGS=FLAGS)

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    model = LSTM_Autoencoder()
    model = model.build_model()
    model.summary()

    if FLAGS.train:
        run.tags = run.tags + ("training",)
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, run=run)
        for _ in trainer.train():
            continue

        trainer.plot_result(ds_train, ds_info["train"]["length"])
        checkpoint = tf.train.Checkpoint(model=model)
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)
    else:
        run.tags = run.tags + ("evaluation",)
        checkpoint = tf.train.Checkpoint(model=model)
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)

if __name__ == "__main__":
    app.run(main)
    