import argparse
import sys
import itertools

class ArgumentParserWithDefaults(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_args = set()

    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        namespace, remaining_args = super().parse_known_args(args, namespace)
        self.explicit_args = {arg[2:] for arg in args if arg.startswith('--')}
        return namespace, remaining_args

    def parse_args(self, args=None, namespace=None):
        namespace, remaining_args = self.parse_known_args(args, namespace)
        if remaining_args:
            msg = 'unrecognized arguments: %s'
            self.error(msg % ' '.join(remaining_args))
        return namespace

        
def get_args():
    parser = ArgumentParserWithDefaults(description="Hyperparameters for SYMPOL RL")

    parser.add_argument(
        "--use_best_config",
        action="store_true",
        help="If true, use the already optimized config from configs.py. This might not exist yet for every environment, in this case the default values are used",
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="If true, render environments each time there is an evaluation",
    )

    parser.add_argument(
        "--overwrite_explicit",
        action="store_true",
    )
    
    parser.add_argument(
        "--adamW",
        action="store_true",
    )

    parser.add_argument(
        "--no-adamW",
        action="store_false",
        dest="adamW",
        help="Do not use AdamW optimizer (explicitly sets to False)"
    )
    
    parser.add_argument(
        "--normEnv",
        action="store_true",
    )
    
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
    )
    
    parser.add_argument(
        "--SWA",
        action="store_true",
    )
    
    parser.add_argument(
        "--dynamic_buffer",
        action="store_true",
        help="Use dynamic trajectory buffer",
    )

    parser.add_argument(
        "--static_batch",
        action="store_true",
        help="Use static batch size",
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--render_each_eval",
        action="store_true",
        help="If true, render environments each time there is an evaluation",
    )
    
    parser.add_argument(
        "--no-render_env",
        dest='render_env',
        action='store_false',
        help="Flag to disable rendering of the environment",
    )
    
    parser.add_argument(
        "--no-reduce_lr",
        dest='reduce_lr',
        action='store_false',
        help="Flag to not use reduce_lr",
    )

    
    
    parser.add_argument(
        "--gpu_number",
        type=int,
        default=0,
        help="GPU Number.",
    )

    parser.add_argument(
        "--optimize_config",
        action="store_true",
        help="If true, use optuna to optimize the parameters specified in the function body of suggest_config in cofigs.py",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials per optuna optimization job.",
    )

    parser.add_argument(
        "--view_size",
        type=int,
        default=3,
        help="MiniGrid view size",
    )    
    # SYMPOL specific parameters
    parser.add_argument("--depth", type=int, default=7, help="Depth for each single estimator / tree")
    parser.add_argument("--n_estimators", type=int, default=1, help="Number of estimators / trees for the ensemble")
    parser.add_argument(
        "--action_type",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Type of the action space, i.e. discrete or continuous (classification vs regression). Continuous actions are not properly implemented yet though and will raise an NotImplementedException.",
    )
    parser.add_argument(
        "--learning_rate_actor",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )    
    parser.add_argument(
        "--learning_rate_actor_weights",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )
    parser.add_argument(
        "--learning_rate_actor_split_values",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )
    parser.add_argument(
        "--learning_rate_actor_split_idx_array",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )
    parser.add_argument(
        "--learning_rate_actor_leaf_array",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )
    parser.add_argument(
        "--learning_rate_actor_log_std",
        type=float,
        default=1e-3,
        help="Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)",
    )
    

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="SDT entmax temperature",
    )    
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of MLP layers",
    )

    parser.add_argument(
        "--neurons_per_layer",
        type=int,
        default=256,
        help="Number of neurons per MLP layer",
    )
    
    # PPO specific parameters
    parser.add_argument(
        "--actor",
        type=str,
        default="sympol",
        choices=["sympol", "mlp", "sdt", "d-sdt", "stateActionDT"],
        help="Specify the actor type: 'sympol' or 'mlp' or 'sdt' 'stateActionDT'",
    )
    parser.add_argument(
        "--critic",
        type=str,
        default="mlp",
        choices=["mlp", "sdt", "sympol"],
        help="Specify the actor type: mlp' or 'sdt'",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="General Advantage Estimator lambda parameter")
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy coefficient, higher value corresponds to more exploration"
    )
    parser.add_argument("--learning_rate_critic", type=float, default=1e-3, help="Learning rate for the critic")
    parser.add_argument(
        "--n_update_epochs", type=int, default=5, help="Number of updates / gradient steps for each batch"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=256,
        help="Number of steps to take in ONE environment before updating the policy / q-approxmation parameters.",
    )
    parser.add_argument("--clip_vloss", action="store_true", help="Clip value function loss")

    parser.add_argument("--clip_coef", type=float, default=0.1, help="Clip coefficient PPO")
    
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument(
        "--accumulate_gradients_every",
        type=int,
        default=1,
        help="Number of accumulation steps for the gradient update. The accumulated gradients will be averaged before backpropagation",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping threshold")
    #parser.add_argument("--target_kl", type=float, default=None, help="Target KL divergence threshold")
    parser.add_argument("--target_kl", type=float, default=None, help="Target KL divergence threshold")
    parser.add_argument("--norm_adv", action="store_true", help="If ture, Normalize the advantages")

    # general parameters
    parser.add_argument(
        "--exp_name", type=str, default="SYMPOL RL", help="Experiment name, important for wandb tracking"
    )
    parser.add_argument("--run_name", type=str, default="Default", help="Name for the current run")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--track", action="store_true", help="If true, initialize a wandb run and track the results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1_000_000,
        help="Number of total environment steps for training. If more than one environment is used, e.g. 5 environments, we have 5 total steps per env.step() call",
    )
    #parser.add_argument(
    #    "--eval_freq", type=int, default=50, help="Frequency of evaluation (in iterations, see argument `n_iteration`)"
    #)
    parser.add_argument(
        "--eval_freq", type=int, default=50_000, help="Frequency of evaluation (in total timesteps)"
    )    
    #parser.add_argument(
    #    "--n_minibatches",
    #    type=int,
    #    default=8,
    #    help="One minibatch is the input that is used for backpropagation / optimization.",
    #)
    
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=128,
        help="Minibatch size used for backpropagation / optimization.",
    )
    
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation. The return of the evaluation is the mean of the cumulative reward across all evaluation episodes",
    )
    parser.add_argument("--n_envs", type=int, default=8, help="Number of environments to use for collecting data")

    parser.add_argument("--random_trials", type=int, default=5, help="Number of random trials for evaluation")
    
    args = parser.parse_args()  
    explicit_args_corrected = []
    for some_arg in parser.explicit_args:
        if 'no-' in some_arg:
            explicit_args_corrected.append(''.join(some_arg.split('no-')))
        else:
            explicit_args_corrected.append(some_arg)
    explicit_arg_values = {arg: getattr(args, arg) for arg in explicit_args_corrected}
    args.__dict__.update(explicit_arg_values)
    if args.use_best_config:
        import configs   

        for name, value in vars(configs).items():
            if 'minigrid' in name and name.split('_')[1] in args.env_id.lower():
                if args.actor == 'stateActionDT':
                    best_cfg = value['mlp']
                elif args.actor == 'd-sdt':
                    best_cfg = value['sdt']
                else:
                    best_cfg = value[args.actor]
                args.__dict__.update(best_cfg)
                break
                
            elif name == '-'.join(args.env_id.lower().split('-')[:-1]):
                if args.actor == 'stateActionDT':
                    best_cfg = value['mlp']
                elif args.actor == 'd-sdt':
                    best_cfg = value['sdt']
                else:
                    best_cfg = value[args.actor]
                args.__dict__.update(best_cfg)
                break
    if args.overwrite_explicit:
        args.__dict__.update(explicit_arg_values) 
          
    return args
