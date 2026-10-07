import os
import gin
import datetime
import wandb
from ast import literal_eval


def gen_run_folder(path_model_id=''):
    '''
    Generates and organizes a folder structure for experiment runs.

        Parameters:
            path_model_id (str, optional): Optional model ID to use in the folder name. Defaults to an empty string.

        Returns:
            dict: Dictionary containing paths to various directories and files used in the experiment.
    '''
    run_paths = dict()

    if not os.path.isdir(path_model_id):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        if path_model_id:
            run_id += '_' + path_model_id
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = path_model_id

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
    #run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval', 'run.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    #run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    '''
    Saves a given configuration string to a file.

        Parameters:
            path_gin (str): Path to the configuration file.
            config (str): Configuration content to be saved.
    '''
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


def config_to_dict(path):
    '''
    Parses a configuration file into a dictionary.

        Parameters:
            path (str): Path to the configuration file.

        Returns:
            dict: A dictionary containing configuration key-value pairs.
    '''
    d = {}
    with open(path) as config:
        for line in config:
            if not line.lstrip().startswith('#') and line.strip():
                line = line.split(" = ")
                d[line[0]] = literal_eval(line[1].rstrip())
    return d


@gin.configurable
def wandb_init(run_paths, project, entity, key, FLAGS):
    '''
    Initializes a Weights & Biases (wandb) run with a given configuration.

        Parameters:
            run_paths (dict): Dictionary containing paths used for logging and configuration.
            project (str): Name of the Weights & Biases project.
            entity (str): Name of the Weights & Biases entity.
            key (str): API key for Weights & Biases authentication.
            FLAGS (FlagValues): Command line flags.
        Returns:
            wandb.run: A wandb run instance.
    '''
    config = {
        "train": FLAGS.train,
        "run-id": FLAGS.run,
    }
    # Define config for wandb run
    config = config_to_dict(run_paths['path_gin'])
    config["train"] = FLAGS.train
    config["run-id"] = run_paths['path_model_id']
    
    if FLAGS.wandb == 'config': # use api key, project and entity from config
        os.environ["WANDB_MODE"] = 'online'
        wandb.login(anonymous='must', key=key)
        run = wandb.init(project=project, entity=entity, config=config)
    elif FLAGS.wandb == '': # use no sync
        os.environ["WANDB_MODE"] = 'offline'
        run = wandb.init(config=config)
    else: # use api key specified in script call
        os.environ["WANDB_MODE"] = 'online'
        wandb.login(anonymous='must', key=FLAGS.wandb)
        run = wandb.init(config=config)
    return run
