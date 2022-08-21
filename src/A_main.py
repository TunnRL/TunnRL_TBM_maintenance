"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Main code that either runs a hyperparameter optimization study with OPTUNA or a
"main run" of just one study with fixed hyperparameters

Created on Sat Oct 30 12:46:42 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""


from joblib import Parallel, delayed
import numpy as np
import optuna
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn  # used in evaluation of yaml file
import warnings
import yaml

from XX_experiment_factory import Optimization, load_best_model
from XX_TBM_environment import CustomEnv
from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter

warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)

###############################################################################
# Constants and fixed variables

# TBM excavation parameters
CUTTERHEAD_RADIUS = 3  # cutterhead radius [m]
TRACK_SPACING = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]
STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode
BROKEN_CUTTERS_THRESH = 0.5  # minimum required % of functional cutters

# MODE determines if either an optimization should run = "Optimization", or a
# new agent is trained with prev. optimized parameters = "Training", or an
# already trained agent is executed = "Execution"
MODE = 'Training'  # 'Optimization', 'Training', 'Execution'

# Checks if env is a suitable gym environment
CHECK_ENV = False

# Parameters for modes "Optimization" and "Training"
EPISODES = 10_000  # max episodes to train for
CHECKPOINT_INTERVAL = 50  # evaluation interval
STUDY = 'SAC_2022_08_21_study'  # name of the study to create / load

# Parameters for "Optimization" mode only
DEFAULT_TRIAL = False  # first run a trial with default parameters.
N_SINGLE_RUN_OPTUNA_TRIALS = 5  # n optuna trials to run in total (including eventual default trial)
N_CORES_PARALLELL = -1  # max number of cores to use for parallel optimization
N_PARALLELL_PROCESSES = 3  # max number of optimizations to run in parallel
MAX_NO_IMPROVEMENT = 2  # maximum number of evaluations without improvement

# Parameters for "Training" mode only
# load best parameters for training from study object or from yaml
# TODO: do an automatic save of parameters upon completing an optuna-experiment session
LOAD_PARAMS_FROM_STUDY = True

# Parameters for "Execution" mode only
EXECUTION_MODEL = "PPO_94f701e8-797d-4834-8462-d76b07436bdf"
NUM_TEST_EPISODES = 1

###############################################################################
# computed/derived variables and instantiations

n_c_tot = int(round((CUTTERHEAD_RADIUS - TRACK_SPACING / 2) / TRACK_SPACING, 0)) + 1
print(f'\ntotal number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), TRACK_SPACING)) - TRACK_SPACING / 2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts,
                CUTTERHEAD_RADIUS, BROKEN_CUTTERS_THRESH)
if CHECK_ENV:
    check_env(env)

agent_name = STUDY.split('_')[0]
assert agent_name in ["PPO", "A2C", "DDPG", "SAC", "TD3"], f"{agent_name} is not a valid agent."

optim = Optimization(n_c_tot, env, STUDY, EPISODES, CHECKPOINT_INTERVAL,
                     MAX_STROKES, agent_name, DEFAULT_TRIAL,
                     MAX_NO_IMPROVEMENT)
hparams = Hyperparameters()
plotter = Plotter()

###############################################################################
# run one of the three modes: Optimization, Training, Execution

if MODE == 'Optimization':  # study
    print(f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} optuna trials are processed\n")

    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler()  # TODO: play around with sampling configs
    study = optuna.create_study(
        direction='maximize', study_name=STUDY, storage=db_file,
        load_if_exists=True, sampler=sampler)

    Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
        delayed(optim.optimize)(N_SINGLE_RUN_OPTUNA_TRIALS) for _ in range(N_PARALLELL_PROCESSES))

    # study = optuna.load_study(study_name=STUDY, storage=db_file)
    # print('Number of finished trials: ', len(study.trials))
    # print('Best trial:')
    # trial = study.best_trial
    # print('  Value: ', trial.value)
    # print('  Params: ')
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")

elif MODE == 'Training':
    print(f'New {agent_name} training run with optimized parameters started.')

    if LOAD_PARAMS_FROM_STUDY is True:
        print(f"loading parameters from the study object: {STUDY}")
        db_path = f"results/{STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=STUDY, storage=db_file)
        best_trial = study.best_trial
        print(f"Highest reward from best trial: {best_trial.value}")

        best_params_dict = hparams.remake_params_dict(
            algorithm=agent_name, raw_params_dict=best_trial.params, env=env,
            n_actions=n_c_tot * n_c_tot)
    else:
        print("loading parameters from yaml file...")
        with open(f"results/algorithm_parameters/{agent_name}.yaml") as file:
            best_params_dict: dict = yaml.safe_load(file)

        best_params_dict["learning_rate"] = hparams.parse_learning_rate(best_params_dict["learning_rate"])    
        best_params_dict["policy_kwargs"] = eval(best_params_dict["policy_kwargs"])
        best_params_dict.update(dict(env=env, n_steps=MAX_STROKES))

    print(f"Parameters used in training: {best_params_dict}")
    optim.train_agent(agent_name=agent_name, best_parameters=best_params_dict)

elif MODE == 'Execution':
    agent_name = EXECUTION_MODEL.split('_')[0]
    agent = load_best_model(agent_name, main_dir="optimization",
                            agent_dir=EXECUTION_MODEL)

    # test agent throughout multiple episodes
    for test_ep_num in range(NUM_TEST_EPISODES):
        print(f"Episode num: {test_ep_num}")
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions = []  # collect actions per episode
        states = [state]  # collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per stroke
        replaced_cutters = []  # collect n of replaced cutters per stroke
        moved_cutters = []  # collect n of moved_cutters cutters per stroke

        # one episode loop
        i = 0
        while not terminal:
            # print(f"Stroke (step) num: {i}")
            # collect number of broken cutters in curr. state
            broken_cutters.append(len(np.where(state == 0)[0]))
            # agent takes an action -> tells which cutters to replace
            action = agent.predict(state, deterministic=False)[0]
            # environment gives new state signal, terminal flag and reward
            state, reward, terminal, info = env.step(action)
            # collect actions, states and rewards for later analyses
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            replaced_cutters.append(env.replaced_cutters)
            moved_cutters.append(env.moved_cutters)
            i += 1

        plotter.state_action_plot(states, actions, n_strokes=200,
                                  n_c_tot=n_c_tot, show=False,
                                  savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg')
        plotter.environment_parameter_plot(test_ep_num, env, show=False,
                                           savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_episode.svg')
        plotter.sample_ep_plot(states, actions, rewards, ep=test_ep_num,
                               savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg',
                               replaced_cutters=replaced_cutters,
                               moved_cutters=moved_cutters)
