# for optuna

#https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
def modify_parameters_by_environment(trial, params, env_id): 
    if False:
        if env_id == 'CartPole-v1':
            params["n_steps"] = trial.suggest_int("n_steps", 16, 64)
            params["n_envs"] = trial.suggest_int("n_envs", 4, 16)          
        elif env_id == 'Pendulum-v1':
            params["n_steps"] = trial.suggest_int("n_steps", 256, 512)
            params["n_envs"] = trial.suggest_int("n_envs", 2, 8)  
        elif env_id == 'BipedalWalker-v3':
            params["n_steps"] = trial.suggest_int("n_steps", 256, 512)
            params["n_envs"] = trial.suggest_int("n_envs", 16, 32)             
        elif env_id == 'LunarLander-v2':
            params["n_steps"] = trial.suggest_int("n_steps", 256, 512)
            params["n_envs"] = trial.suggest_int("n_envs", 8, 16)      
    else:
        #params["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512]) #trial.suggest_int("n_steps", 128, 512)
        params["n_steps"] = trial.suggest_categorical("n_steps", [128, 512]) #trial.suggest_int("n_steps", 128, 512)
        params["n_envs"] = trial.suggest_int("n_envs", 4, 16)      
    return params

    
def suggest_config_sympol(trial, env_id=''):
    params = {
        "learning_rate_actor_weights": trial.suggest_float("learning_rate_actor_weights", 0.0001, 0.1, log=True),
        "learning_rate_actor_split_values": trial.suggest_float("learning_rate_actor_split_values", 0.0001, 0.05, log=True),
        "learning_rate_actor_split_idx_array": trial.suggest_float("learning_rate_actor_split_idx_array", 0.0001, 0.1, log=True),
        "learning_rate_actor_leaf_array": trial.suggest_float("learning_rate_actor_leaf_array", 0.0001, 0.05, log=True),
        "learning_rate_actor_log_std": trial.suggest_float("learning_rate_actor_log_std", 0.0001, 0.1, log=True),
        
        "learning_rate_critic": trial.suggest_float("learning_rate_critic", 0.0001, 0.01, log=True),
        
        "SWA": trial.suggest_categorical("SWA", [True]), #FALSE | TRUE
        "adamW": trial.suggest_categorical("adamW", [True]), #TRUE
        "reduce_lr": trial.suggest_categorical("reduce_lr", [True, False]), #FALSE | TRUE
        
        "dropout": trial.suggest_categorical("dropout", [0.0]), #IMPLEMENT
        #"depth": trial.suggest_int("depth", 9),
        "depth": trial.suggest_categorical("depth", [7]),
        
        #"n_minibatches": trial.suggest_int("n_minibatches", 1, 16),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64]),
        #"accumulate_gradients_every": trial.suggest_categorical("accumulate_gradients_every", [1,1]),
        
        "n_update_epochs": trial.suggest_int("n_update_epochs", 1, 10),
        #"max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 1000]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [1000]),
        "norm_adv": trial.suggest_categorical("norm_adv", [True, False]),
        
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.1, 0.2, 0.5]),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75]),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.999]), #no 0.8
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.95, 0.99]), #INCLUDE
        
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
        #"n_estimators": trial.suggest_int("n_estimators", 10, 100),
        
    }
    params = modify_parameters_by_environment(trial, params, env_id)
 
    
    return params

def suggest_config_mlp(trial, env_id=''):
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "neurons_per_layer": trial.suggest_int("neurons_per_layer", 16, 256),
        
        "learning_rate_actor": trial.suggest_float("learning_rate_actor", 0.0001, 0.01, log=True),
        "learning_rate_critic": trial.suggest_float("learning_rate_critic", 0.0001, 0.01, log=True),
        "reduce_lr": trial.suggest_categorical("reduce_lr", [False]),
        "adamW": trial.suggest_categorical("adamW", [False]),
        
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256, 512]),
        
        "n_update_epochs": trial.suggest_int("n_update_epochs", 1, 10),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 1000]),
        "norm_adv": trial.suggest_categorical("norm_adv", [True, False]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.1, 0.2]),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75]),       

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.99]),      
    }
    params = modify_parameters_by_environment(trial, params, env_id)
     
    return params

def suggest_config_sdt(trial, env_id=''):
    params = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate_actor": trial.suggest_float("learning_rate_actor", 0.0001, 0.01, log=True),
        "learning_rate_critic": trial.suggest_float("learning_rate_critic", 0.0001, 0.01, log=True),
        "reduce_lr": trial.suggest_categorical("reduce_lr", [False]),
        "adamW": trial.suggest_categorical("adamW", [False]),        
        "temperature": trial.suggest_categorical("temperature", [0.01, 0.05, 0.1, 0.5, 1, 1, 1, 1]),

        "critic": trial.suggest_categorical("critic", ["mlp", "sdt"]),
        
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256, 512]),
        
        "n_update_epochs": trial.suggest_int("n_update_epochs", 1, 10),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 1000]),
        "norm_adv": trial.suggest_categorical("norm_adv", [True, False]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.1, 0.2]),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75]),       

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.99]),          
    }
    params = modify_parameters_by_environment(trial, params, env_id)
 
    
    return params


    
def suggest_config_dsdt(trial, env_id=''):
    params = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate_actor": trial.suggest_float("learning_rate_actor", 0.0001, 0.01, log=True),
        "learning_rate_critic": trial.suggest_float("learning_rate_critic", 0.0001, 0.01, log=True),
        "reduce_lr": trial.suggest_categorical("reduce_lr", [False]),
        "adamW": trial.suggest_categorical("adamW", [False]),        
        "temperature": trial.suggest_categorical("temperature", [0.01, 0.05, 0.1, 0.5, 1, 1, 1, 1]),

        "critic": trial.suggest_categorical("critic", ["mlp", "sdt"]),
        
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256, 512]),
        
        "n_update_epochs": trial.suggest_int("n_update_epochs", 1, 10),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 1000]),
        "norm_adv": trial.suggest_categorical("norm_adv", [True, False]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.1, 0.2]),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75]),       

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.99]),              
    }

    params = modify_parameters_by_environment(trial, params, env_id)

    return params

def suggest_config_stateActionDT(trial, env_id=''):
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "neurons_per_layer": trial.suggest_int("neurons_per_layer", 16, 256),
        
        "learning_rate_actor": trial.suggest_float("learning_rate_actor", 0.0001, 0.01, log=True),
        "learning_rate_critic": trial.suggest_float("learning_rate_critic", 0.0001, 0.01, log=True),
        "reduce_lr": trial.suggest_categorical("reduce_lr", [False]),
        "adamW": trial.suggest_categorical("adamW", [False]),
        
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256, 512]),
        
        "n_update_epochs": trial.suggest_int("n_update_epochs", 1, 10),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 1000]),
        "norm_adv": trial.suggest_categorical("norm_adv", [True, False]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.1, 0.2]),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75]),       

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.99]),      
    }
    params = modify_parameters_by_environment(trial, params, env_id)

    
    return params
    

cartpole = {
    'mlp': { 
            "adamW": False,

            "ent_coef": 0.2,

            "gae_lambda": 0.9,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.00142,
            "learning_rate_critic": 0.002765,

            "max_grad_norm": 1.0,
            "minibatch_size": 256,
        
            "n_envs": 13,
            "n_steps": 128,
        
            "n_update_epochs": 7,
            "neurons_per_layer": 139,
            "norm_adv": False,
            "num_layers": 2,
            
            "reduce_lr": False,

            "vf_coef": 0.25,
        },
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 7,
            "ent_coef": 0.0,

            "gae_lambda": 0.95,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0009255,
            "learning_rate_critic": 0.0001238,

            "max_grad_norm": 0.1,
            "minibatch_size": 128,
        
            "n_envs": 15,
            "n_steps": 512,
        
            "n_update_epochs": 4,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.50,
        },
    'sympol': {
        "ent_coef": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    
        "learning_rate_actor_weights": 0.04760308580677977,
        "learning_rate_actor_split_values": 0.000222274485191996,
        "learning_rate_actor_split_idx_array": 0.025528008432059508,
        "learning_rate_actor_leaf_array": 0.019530943718321373,
        "learning_rate_actor_log_std": 0.0012313062437960766,
        
        "learning_rate_critic": 0.0013329992676131342,

        "max_grad_norm": 1000,
        "n_envs": 7,
        "n_steps": 512,
        "n_update_epochs": 7,
        "norm_adv": False,
        "reduce_lr": True,
        "vf_coef": 0.50,

    
        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },    
    
}


pendulum = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0003681,
            "learning_rate_critic": 0.001936,

            "max_grad_norm": 1000.0,
            "minibatch_size": 128,
        
            "n_envs": 8,
            "n_steps": 512,
        
            "n_update_epochs": 2,
            "neurons_per_layer": 75,
            "norm_adv": True,
            "num_layers": 2,
            
            "reduce_lr": False,

            "vf_coef": 0.25,
        },
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 7,
            "ent_coef": 0.2,

            "gae_lambda": 0.9,
            "gamma": 0.9,
        
            "learning_rate_actor": 0.000364,
            "learning_rate_critic": 0.0001127,

            "max_grad_norm": 0.1,
            "minibatch_size": 128,
        
            "n_envs": 7,
            "n_steps": 256,
        
            "n_update_epochs": 7,
            "norm_adv": False,
            
            "reduce_lr": False,

            "temperature": 0.1,
        
            "vf_coef": 0.50,
        },
    'sympol': {
        "ent_coef": 0.1,
        "gae_lambda": 0.8,
        "gamma": 0.999,
    
        "learning_rate_actor_weights": 0.021704708446679,
        "learning_rate_actor_split_values": 0.0002307526719494789,
        "learning_rate_actor_split_idx_array": 0.009862044169880627,
        "learning_rate_actor_leaf_array": 0.006414075616512551,
        "learning_rate_actor_log_std": 0.00015395109187787975,
        
        "learning_rate_critic": 0.00032866087550350426,

        "max_grad_norm": 1000.0,
        "n_envs": 15,
        "n_steps": 128,
        "n_update_epochs": 7,
        "norm_adv": True,    
        "reduce_lr": False,
        "vf_coef": 0.75,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}


mountaincar = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.9,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.0001266,
            "learning_rate_critic": 0.007337,

            "max_grad_norm": 1000.0,
            "minibatch_size": 512,
        
            "n_envs": 10,
            "n_steps": 128,
        
            "n_update_epochs": 6,
            "neurons_per_layer": 144,
            "norm_adv": True,
            "num_layers": 3,
            
            "reduce_lr": False,

            "vf_coef": 0.25,
        },
    'sdt': {
            "adamW": False,

            "critic": 'sdt',
            "depth": 8,
            "ent_coef": 0.0,

            "gae_lambda": 0.9,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.003937312675632405,
            "learning_rate_critic": 0.002474020951866176,

            "max_grad_norm": 0.5,
            "minibatch_size": 64,
        
            "n_envs": 4,
            "n_steps": 128,
        
            "n_update_epochs": 3,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },
    'sympol': {
        "ent_coef": 0.2,
        "gae_lambda": 0.8,
        "gamma": 0.95,
    
        "learning_rate_actor_weights": 0.01387037630070651,
        "learning_rate_actor_split_values": 0.0003476933014461969,
        "learning_rate_actor_split_idx_array": 0.0001389227923335404,
        "learning_rate_actor_leaf_array": 0.0020166491255254617,
        "learning_rate_actor_log_std": 0.00041038262325912274,
        
        "learning_rate_critic": 0.0007584969721345409,

        "max_grad_norm": 1000.0,
        "n_envs": 9,
        "n_steps": 128,
        "n_update_epochs": 6,
        "norm_adv": False,    
        "reduce_lr": True,
        "vf_coef": 0.75,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}

mountaincarcontinuous = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.004786,
            "learning_rate_critic": 0.001164,

            "max_grad_norm": 0.1,
            "minibatch_size": 512,
        
            "n_envs": 15,
            "n_steps": 512,
        
            "n_update_epochs": 2,
            "neurons_per_layer": 240,
            "norm_adv": True,
            "num_layers": 2,
            
            "reduce_lr": False,

            "vf_coef": 0.25,
        },
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 7,
            "ent_coef": 0.0,

            "gae_lambda": 0.9,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0008297,
            "learning_rate_critic": 0.007393,

            "max_grad_norm": 0.5,
            "minibatch_size": 64,
        
            "n_envs": 14,
            "n_steps": 512,
        
            "n_update_epochs": 1,
            "norm_adv": False,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.25,
        },
    'sympol': {
        "ent_coef": 0.5,
        "gae_lambda": 0.99,
        "gamma": 0.999,
    
        "learning_rate_actor_weights": 0.00012314339663197326,
        "learning_rate_actor_split_values": 0.0001160748504514767,
        "learning_rate_actor_split_idx_array": 0.0001015527526014825,
        "learning_rate_actor_leaf_array": 0.028465599628829257,
        "learning_rate_actor_log_std": 0.09429967607128892,
        
        "learning_rate_critic": 0.0020613382527496695,

        "max_grad_norm": 1000.0,
        "n_envs": 5,
        "n_steps": 128,
        "n_update_epochs": 2,
        "norm_adv": False,    
        "reduce_lr": True,
        "vf_coef": 0.5,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}


acrobot = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.0,

            "gae_lambda": 0.9,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0002193,
            "learning_rate_critic": 0.004594,

            "max_grad_norm": 1.0,
            "minibatch_size": 256,
        
            "n_envs": 12,
            "n_steps": 512,
        
            "n_update_epochs": 9,
            "neurons_per_layer": 185,
            "norm_adv": True,
            "num_layers": 2,
            
            "reduce_lr": False,

            "vf_coef": 0.50,
        },
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 6,
            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0016799276982439016,
            "learning_rate_critic": 0.0003204461305956749,

            "max_grad_norm": 0.1,
            "minibatch_size": 128,
        
            "n_envs": 6,
            "n_steps": 128,
        
            "n_update_epochs": 10,
            "norm_adv": False,
            
            "reduce_lr": False,

            "temperature": 0.5,
        
            "vf_coef": 0.50,
        },
    'sympol': {
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    
        "learning_rate_actor_weights": 0.002736155180804038,
        "learning_rate_actor_split_values": 0.00020085566411900057,
        "learning_rate_actor_split_idx_array": 0.05198040198477529,
        "learning_rate_actor_leaf_array": 0.005371878728382642,
        "learning_rate_actor_log_std": 0.0019814944246277504,
        
        "learning_rate_critic": 0.0003547997953897775,

        "max_grad_norm": 1000.0,
        "n_envs": 8,
        "n_steps": 128,
        "n_update_epochs": 7,
        "norm_adv": False,    
        "reduce_lr": True,
        "vf_coef": 0.25,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
}

lunarlander = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.9,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.0005870197711720902,
            "learning_rate_critic": 0.0032635608663862353,

            "max_grad_norm": 0.5,
            "minibatch_size": 128,
        
            "n_envs": 13,
            "n_steps": 512,
        
            "n_update_epochs": 8,
            "neurons_per_layer": 46,
            "norm_adv": False,
            "num_layers": 3,
            
            "reduce_lr": False,

            "vf_coef": 0.50,
        },
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 8,
            "ent_coef": 0.2,

            "gae_lambda": 0.99,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.0006108017244234425,
            "learning_rate_critic": 0.0011201875111956177,

            "max_grad_norm": 1.0,
            "minibatch_size": 128,
        
            "n_envs": 7,
            "n_steps": 512,
        
            "n_update_epochs": 2,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },
    'sympol': {
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "gamma": 0.999,
    
        "learning_rate_actor_weights": 0.07157685619992765,
        "learning_rate_actor_split_values": 0.0006591868973696417,
        "learning_rate_actor_split_idx_array": 0.009966850522393832,
        "learning_rate_actor_leaf_array": 0.008588600717840487,
        "learning_rate_actor_log_std": 0.02140711489067244,
        
        "learning_rate_critic": 0.001771755240346081,

        "max_grad_norm": 1000.0,
        "n_envs": 6,
        "n_steps": 512,
        "n_update_epochs": 7,
        "norm_adv": True,    
        "reduce_lr": True,
        "vf_coef": 0.5,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}


minigrid_lavagaps5 = {

    'mlp': { #LavaGapS5
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.001808,
            "learning_rate_critic": 0.00304,

            "max_grad_norm": 1,
            "minibatch_size": 128,
        
            "n_envs": 8,
            "n_steps": 512,
        
            "n_update_epochs": 9,
            "neurons_per_layer": 76,
            "norm_adv": False,
            "num_layers": 1,
            
            "reduce_lr": False,

            "vf_coef": 0.25,
        },    
    'sdt': { #LavaGapS5
            "adamW": False,

            "critic": 'sdt',
            "depth": 7,
            "ent_coef": 0.2,

            "gae_lambda": 0.99,
            "gamma": 0.999,
        
            "learning_rate_actor": 0.0004584,
            "learning_rate_critic": 0.0002554,

            "max_grad_norm": 0.5,
            "minibatch_size": 512,
        
            "n_envs": 10,
            "n_steps": 256,
        
            "n_update_epochs": 8,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },

    'sympol': {
        "ent_coef": 0.1,
        "gae_lambda": 0.9,
        "gamma": 0.95,
    
        "learning_rate_actor_weights": 0.054914151484351845,
        "learning_rate_actor_split_values": 0.005811380648459824,
        "learning_rate_actor_split_idx_array": 0.01225369992015828,
        "learning_rate_actor_leaf_array": 0.008676695448646759,
        "learning_rate_actor_log_std": 0.004742570909023367,
        
        "learning_rate_critic": 0.0006092740766519476,

        "max_grad_norm": 1000.0,
        "n_envs": 16,
        "n_steps": 512,
        "n_update_epochs": 5,
        "norm_adv": True,    
        "reduce_lr": True,
        "vf_coef": 0.25,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
       
}

minigrid_doorkey = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.9,
            "gamma": 0.9,
        
            "learning_rate_actor": 0.0004126,
            "learning_rate_critic": 0.0004508,

            "max_grad_norm": 0.1,
            "minibatch_size": 256,
        
            "n_envs": 8,
            "n_steps": 256,
        
            "n_update_epochs": 7,
            "neurons_per_layer": 169,
            "norm_adv": True,
            "num_layers": 1,
            
            "reduce_lr": False,

            "vf_coef": 0.5,
        },
    
    'sdt': {
            "adamW": False,

            "critic": 'mlp',
            "depth": 6,
            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.9,
        
            "learning_rate_actor": 0.0008919,
            "learning_rate_critic": 0.001508,

            "max_grad_norm": 0.1,
            "minibatch_size": 256,
        
            "n_envs": 10,
            "n_steps": 256,
        
            "n_update_epochs": 10,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },

    'sympol': {
        "ent_coef": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    
        "learning_rate_actor_weights": 0.04168761929778069,
        "learning_rate_actor_split_values": 0.0012450034110784152,
        "learning_rate_actor_split_idx_array": 0.0005029536099891734,
        "learning_rate_actor_leaf_array": 0.0035778989299146166,
        "learning_rate_actor_log_std": 0.02121882708532112,
        
        "learning_rate_critic": 0.0008539673613239264,

        "max_grad_norm": 1000.0,
        "n_envs": 14,
        "n_steps": 512,
        "n_update_epochs": 9,
        "norm_adv": True,    
        "reduce_lr": True,
        "vf_coef": 0.50,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}

minigrid_empty = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.00022006113628729703,
            "learning_rate_critic": 0.001016745520591446,

            "max_grad_norm": 0.1,
            "minibatch_size": 64,
        
            "n_envs": 13,
            "n_steps": 512,
        
            "n_update_epochs": 5,
            "neurons_per_layer": 112,
            "norm_adv": False,
            "num_layers": 3,
            
            "reduce_lr": False,

            "vf_coef": 0.5,
        },
    
    'sdt': {
            "adamW": False,

            "critic": 'sdt',
            "depth": 7,
            "ent_coef": 0.1,

            "gae_lambda": 0.9,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.004439821550585952,
            "learning_rate_critic": 0.0004488603528109438,

            "max_grad_norm": 0.1,
            "minibatch_size": 512,
        
            "n_envs": 10,
            "n_steps": 512,
        
            "n_update_epochs": 5,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },

    'sympol': {
        "ent_coef": 0.1,
        "gae_lambda": 0.99,
        "gamma": 0.9,
    
        "learning_rate_actor_weights": 0.06344895603031736,
        "learning_rate_actor_split_values": 0.0009245327872865724,
        "learning_rate_actor_split_idx_array": 0.0006304100125886239,
        "learning_rate_actor_leaf_array": 0.002646152248961896,
        "learning_rate_actor_log_std": 0.04341143118701816,
        
        "learning_rate_critic": 0.0006610280521337505,

        "max_grad_norm": 1000.0,
        "n_envs": 14,
        "n_steps": 128,
        "n_update_epochs": 8,
        "norm_adv": True,    
        "reduce_lr": False,
        "vf_coef": 0.50,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}

minigrid_lavagaps7 = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.9,
        
            "learning_rate_actor": 0.00030421380932650844,
            "learning_rate_critic": 0.0005672446115512332,

            "max_grad_norm": 0.5,
            "minibatch_size": 512,
        
            "n_envs": 12,
            "n_steps": 128,
        
            "n_update_epochs": 8,
            "neurons_per_layer": 28,
            "norm_adv": True,
            "num_layers": 1,
            
            "reduce_lr": False,

            "vf_coef": 0.75,
        },
    
    'sdt': {
            "adamW": False,

            "critic": 'sdt',
            "depth": 8,
            "ent_coef": 0.1,

            "gae_lambda": 0.95,
            "gamma": 0.95,
        
            "learning_rate_actor": 0.0019084174917437576,
            "learning_rate_critic": 0.00510744773741252,

            "max_grad_norm": 0.1,
            "minibatch_size": 256,
        
            "n_envs": 13,
            "n_steps": 128,
        
            "n_update_epochs": 4,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.25,
        },

    'sympol': {
        "ent_coef": 0.1,
        "gae_lambda": 0.9,
        "gamma": 0.99,
    
        "learning_rate_actor_weights": 0.0005793722871267403,
        "learning_rate_actor_split_values": 0.0005838223729862216,
        "learning_rate_actor_split_idx_array": 0.0006590714932633344,
        "learning_rate_actor_leaf_array": 0.007946523254059177,
        "learning_rate_actor_log_std": 0.002205830616639246,
        
        "learning_rate_critic": 0.001127757835458702,

        "max_grad_norm": 1000.0,
        "n_envs": 7,
        "n_steps": 128,
        "n_update_epochs": 4,
        "norm_adv": True,    
        "reduce_lr": True,
        "vf_coef": 0.5,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}

minigrid_distshift1 = {
    'mlp': {
            "adamW": False,

            "ent_coef": 0.1,

            "gae_lambda": 0.99,
            "gamma": 0.99,
        
            "learning_rate_actor": 0.0002579884429826318,
            "learning_rate_critic": 0.000950682421442667,

            "max_grad_norm": 0.1,
            "minibatch_size": 256,
        
            "n_envs": 10,
            "n_steps": 128,
        
            "n_update_epochs": 7,
            "neurons_per_layer": 158,
            "norm_adv": True,
            "num_layers": 2,
            
            "reduce_lr": False,

            "vf_coef": 0.5,
        },
    
    'sdt': {
            "adamW": False,

            "critic": 'sdt',
            "depth": 7,
            "ent_coef": 0.1,

            "gae_lambda": 0.9,
            "gamma": 0.95,
        
            "learning_rate_actor": 0.0008335968492146473,
            "learning_rate_critic": 0.0020439708107133216,

            "max_grad_norm": 1000.0,
            "minibatch_size": 512,
        
            "n_envs": 5,
            "n_steps": 512,
        
            "n_update_epochs": 7,
            "norm_adv": True,
            
            "reduce_lr": False,

            "temperature": 1,
        
            "vf_coef": 0.75,
        },

    'sympol': {
        "ent_coef": 0.5,
        "gae_lambda": 0.95,
        "gamma": 0.999,
    
        "learning_rate_actor_weights":0.03580088868256987,
        "learning_rate_actor_split_values": 0.0002680425031090237,
        "learning_rate_actor_split_idx_array": 0.008701058712472901,
        "learning_rate_actor_leaf_array": 0.0005740321057491008,
        "learning_rate_actor_log_std": 0.03767558661659253,
        
        "learning_rate_critic": 0.0009300326937305064,

        "max_grad_norm": 1000.0,
        "n_envs": 10,
        "n_steps": 512,
        "n_update_epochs": 5,
        "norm_adv": False,    
        "reduce_lr": True,
        "vf_coef": 0.25,

        "SWA": True,
        "adamW": True,  
        "dropout": 0.0,
        "depth": 7,       
        "minibatch_size": 64,
        "n_estimators": 1,    
        #"clip_coef": trial.suggest_float("clip_coef", 0.1, 0.5),
    },  
    
}
