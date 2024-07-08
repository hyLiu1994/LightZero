import torch
import torch.optim as optim
import random
from auto_LiRPA import BoundedModule
from auto_LiRPA.eps_scheduler import LinearScheduler
from .models import *
from .logging import *
import os
import sys
class Trainer(object):
    '''
    This is a class representing a Policy Gradient trainer, which 
    trains both a deep Policy network and a deep Value network.
    Exposes functions:
    - advantage_and_return
    - multi_actor_step
    - reset_envs
    - run_trajectories
    - train_step
    Trainer also handles all logging, which is done via the "cox"
    library
    '''
    def __init__(self, policy_net_class, value_net_class, params,
                 store, advanced_logging=True, log_every=5):
        '''
        Initializes a new Trainer class.
        Inputs;
        - policy, the class of policy network to use (inheriting from nn.Module)
        - val, the class of value network to use (inheriting from nn.Module)
        - step, a reference to a function to use for the policy step (see steps.py)
        - params, an dictionary with all of the required hyperparameters
        '''
        # Parameter Loading
        self.params = Parameters(params)

        # Whether or not the value network uses the current timestep
        time_in_state = self.VALUE_CALC == "time"

        # Whether to use GPU (as opposed to CPU)
        if not self.CPU:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.envs = None
        self.params.AGENT_TYPE = params['agent_type']
        self.params.NUM_ACTIONS = params['num_actions']
        self.params.NUM_FEATURES = params['num_features']
        self.params.MAX_KL_INCREMENT = (self.params.MAX_KL_FINAL - self.params.MAX_KL) / self.params.TRAIN_STEPS
        self.advanced_logging = advanced_logging
        self.n_steps = 0
        self.log_every = log_every
        self.policy_net_class = policy_net_class

        # Instantiation
        self.policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS,
                                             self.INITIALIZATION,
                                             time_in_state=time_in_state,
                                             activation=self.policy_activation)

        # Instantiate convex relaxation model when mode is 'robust_ppo'
        if self.MODE == 'robust_ppo' or self.MODE == 'adv_sa_ppo':
            self.create_relaxed_model(time_in_state)

        # Minimax training
        if self.MODE == 'adv_ppo' or self.MODE == 'adv_trpo' or self.MODE == 'adv_sa_ppo':
            # Copy parameters if they are set to "same".
            if self.params.ADV_PPO_LR_ADAM == "same":
                self.params.ADV_PPO_LR_ADAM = self.params.PPO_LR_ADAM
            if self.params.ADV_VAL_LR == "same":
                self.params.ADV_VAL_LR = self.params.VAL_LR
            if self.params.ADV_CLIP_EPS == "same":
                self.params.ADV_CLIP_EPS = self.params.CLIP_EPS
            if self.params.ADV_EPS == "same":
                self.params.ADV_EPS = self.params.ROBUST_PPO_EPS
            if self.params.ADV_ENTROPY_COEFF == "same":
                self.params.ADV_ENTROPY_COEFF = self.params.ENTROPY_COEFF
            # The adversary policy has features as input, features as output.
            self.adversary_policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=time_in_state,
                                                 activation=self.policy_activation)
            # Optimizer for adversary
            self.params.ADV_POLICY_ADAM = optim.Adam(self.adversary_policy_model.parameters(), lr=self.ADV_PPO_LR_ADAM, eps=1e-5)

            # Adversary value function.
            self.adversary_val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
            self.adversary_val_opt = optim.Adam(self.adversary_val_model.parameters(), lr=self.ADV_VAL_LR, eps=1e-5)
            assert self.adversary_policy_model.discrete == (self.AGENT_TYPE == "discrete")

            # Learning rate annealling for adversary.
            if self.ANNEAL_LR:
                adv_lam = lambda f: 1-f/self.TRAIN_STEPS
                adv_ps = optim.lr_scheduler.LambdaLR(self.ADV_POLICY_ADAM, 
                                                        lr_lambda=adv_lam)
                adv_vs = optim.lr_scheduler.LambdaLR(self.adversary_val_opt, lr_lambda=adv_lam)
                self.params.ADV_POLICY_SCHEDULER = adv_ps
                self.params.ADV_VALUE_SCHEDULER = adv_vs

        opts_ok = (self.PPO_LR == -1 or self.PPO_LR_ADAM == -1)
        assert opts_ok, "One of ppo_lr and ppo_lr_adam must be -1 (off)."
        # Whether we should use Adam or simple GD to optimize the policy parameters
        if self.PPO_LR_ADAM != -1:
            kwargs = {
                'lr':self.PPO_LR_ADAM,
            }

            if self.params.ADAM_EPS > 0:
                kwargs['eps'] = self.ADAM_EPS

            self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(),
                                                 **kwargs)
        else:
            self.params.POLICY_ADAM = optim.SGD(self.policy_model.parameters(), lr=self.PPO_LR)

        # If using a time dependent value function, add one extra feature
        # for the time ratio t/T
        if time_in_state:
            self.params.NUM_FEATURES = self.NUM_FEATURES + 1

        # Value function optimization
        self.val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
        self.val_opt = optim.Adam(self.val_model.parameters(), lr=self.VAL_LR, eps=1e-5) 
        assert self.policy_model.discrete == (self.AGENT_TYPE == "discrete")

        # Learning rate annealing
        # From OpenAI hyperparametrs:
        # Set adam learning rate to 3e-4 * alpha, where alpha decays from 1 to 0 over training
        if self.ANNEAL_LR:
            lam = lambda f: 1-f/self.TRAIN_STEPS
            ps = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, 
                                                    lr_lambda=lam)
            vs = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)
            self.params.POLICY_SCHEDULER = ps
            self.params.VALUE_SCHEDULER = vs

        if store is not None:
            self.setup_stores(store)
        else:
            print("Not saving results to cox store.")

    

    def create_relaxed_model(self, time_in_state=False):
        # Create state perturbation model for robust PPO training.
        if isinstance(self.policy_model, CtsPolicy):
            if self.ROBUST_PPO_METHOD == "convex-relax":
                from .convex_relaxation import RelaxedCtsPolicyForState
                relaxed_policy_model = RelaxedCtsPolicyForState(
                        self.NUM_FEATURES, self.NUM_ACTIONS, time_in_state=time_in_state,
                        activation=self.policy_activation, policy_model=self.policy_model)
                dummy_input1 = torch.randn(1, self.NUM_FEATURES)
                inputs = (dummy_input1, )
                self.relaxed_policy_model = BoundedModule(relaxed_policy_model, inputs)
            else:
                # For SGLD no need to create the relaxed model
                self.relaxed_policy_model = None
            self.robust_eps_scheduler = LinearScheduler(self.params.ROBUST_PPO_EPS, self.params.ROBUST_PPO_EPS_SCHEDULER_OPTS)
            if self.params.ROBUST_PPO_BETA_SCHEDULER_OPTS == "same":
                self.robust_beta_scheduler = LinearScheduler(self.params.ROBUST_PPO_BETA, self.params.ROBUST_PPO_EPS_SCHEDULER_OPTS)
            else:
                self.robust_beta_scheduler = LinearScheduler(self.params.ROBUST_PPO_BETA, self.params.ROBUST_PPO_BETA_SCHEDULER_OPTS)
        else:
            raise NotImplementedError

    def setup_stores(self, store):
        # Logging setup
        self.store = store
        if self.MODE == 'adv_ppo' or self.MODE == 'adv_trpo' or self.MODE == 'adv_sa_ppo':
            adv_optimization_table = {
                'mean_reward':float,
                'final_value_loss':float,
                'final_policy_loss':float,
                'final_surrogate_loss':float,
                'entropy_bonus':float,
                'mean_std':float
            }
            self.store.add_table('optimization_adv', adv_optimization_table)
        optimization_table = {
            'mean_reward':float,
            'final_value_loss':float,
            'final_policy_loss':float,
            'final_surrogate_loss':float,
            'entropy_bonus':float,
            'mean_std':float,
        }
        self.store.add_table('optimization', optimization_table)

        if self.advanced_logging:
            paper_constraint_cols = {
                'avg_kl':float,
                'max_kl':float,
                'max_ratio':float,
                'opt_step':int
            }

            value_cols = {
                'heldout_gae_loss':float,
                'heldout_returns_loss':float,
                'train_gae_loss':float,
                'train_returns_loss':float
            }

            weight_cols = {}
            for name, _ in self.policy_model.named_parameters():
                name += "."
                for k in ["l1", "l2", "linf", "delta_l1", "delta_l2", "delta_linf"]:
                    weight_cols[name + k] = float

            self.store.add_table('paper_constraints_train',
                                        paper_constraint_cols)
            self.store.add_table('paper_constraints_heldout',
                                        paper_constraint_cols)
            self.store.add_table('value_data', value_cols)
            self.store.add_table('weight_updates', weight_cols)

        if self.params.MODE == 'robust_ppo' or self.params.MODE == 'adv_sa_ppo':
            robust_cols ={
                'eps': float,
                'beta': float,
                'kl': float,
                'surrogate': float,
                'entropy': float,
                'loss': float,
            }
            self.store.add_table('robust_ppo_data', robust_cols)


    def __getattr__(self, x):
        '''
        Allows accessing self.A instead of self.params.A
        '''
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)

    """Conduct adversarial attack using value network."""
    def apply_attack(self, last_states):
        if self.params.ATTACK_RATIO < random.random():
            # Only attack a portion of steps.
            return last_states
        eps = self.params.ATTACK_EPS
        if eps == "same":
            eps = self.params.ROBUST_PPO_EPS
        else:
            eps = float(eps)
        steps = self.params.ATTACK_STEPS
        if self.params.ATTACK_METHOD == "critic":
            # Find a state that is close the last_states and decreases value most.
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # Random start.
                noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                states = last_states + noise
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        value = self.val_model(states).mean(dim=1)
                        value.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "random":
            # Apply an uniform random noise.
            noise = torch.empty_like(last_states).uniform_(-eps, eps)
            return (last_states + noise).detach()
        elif self.params.ATTACK_METHOD == "action" or self.params.ATTACK_METHOD == "action+imit":
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # SGLD noise factor. We simply set beta=1.
                noise_factor = np.sqrt(2 * step_eps)
                noise = torch.randn_like(last_states) * noise_factor
                # The first step has gradient zero, so add the noise and projection directly.
                states = last_states + noise.sign() * step_eps
                # Current action at this state.
                if self.params.ATTACK_METHOD == "action+imit":
                    if not hasattr(self, "imit_network") or self.imit_network == None:
                        assert self.params.imit_model_path != None
                        print('\nLoading imitation network for attack: ', self.params.imit_model_path)
                        # Setup imitation network
                        self.setup_imit(train=False)
                        imit_ckpt = torch.load(self.params.imit_model_path)
                        self.imit_network.load_state_dict(imit_ckpt['state_dict'])
                        self.imit_network.reset()
                        self.imit_network.pause_history()
                    old_action, old_stdev = self.imit_network(last_states)
                else:
                    old_action, old_stdev = self.policy_model(last_states)
                # Normalize stdev, avoid numerical issue
                old_stdev /= (old_stdev.mean())
                old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        if self.params.ATTACK_METHOD == "action+imit":
                            action_change = (self.imit_network(states)[0] - old_action) / old_stdev
                        else:
                            action_change = (self.policy_model(states)[0] - old_action) / old_stdev
                        action_change = (action_change * action_change).sum(dim=1)
                        action_change.backward()
                        # Reduce noise at every step.
                        noise_factor = np.sqrt(2 * step_eps) / (i+2)
                        # Project noisy gradient to step boundary.
                        update = (states.grad + noise_factor * torch.randn_like(last_states)).sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data + update, clamp_min), clamp_max)
                    if self.params.ATTACK_METHOD == "action+imit": 
                        self.imit_network.zero_grad() 
                    self.policy_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "sarsa" or self.params.ATTACK_METHOD == "sarsa+action":
            # Attack using a learned value network.
            assert self.params.ATTACK_SARSA_NETWORK is not None
            use_action = self.params.ATTACK_SARSA_ACTION_RATIO > 0 and self.params.ATTACK_METHOD == "sarsa+action"
            action_ratio = self.params.ATTACK_SARSA_ACTION_RATIO
            assert action_ratio >= 0 and action_ratio <= 1
            if not hasattr(self, "sarsa_network"):
                self.sarsa_network = ValueDenseNet(state_dim=self.NUM_FEATURES+self.NUM_ACTIONS, init="normal")
                print("Loading sarsa network", self.params.ATTACK_SARSA_NETWORK)
                sarsa_ckpt = torch.load(self.params.ATTACK_SARSA_NETWORK)
                sarsa_meta = sarsa_ckpt['metadata']
                sarsa_eps = sarsa_meta['sarsa_eps'] if 'sarsa_eps' in sarsa_meta else "unknown"
                sarsa_reg = sarsa_meta['sarsa_reg'] if 'sarsa_reg' in sarsa_meta else "unknown"
                sarsa_steps = sarsa_meta['sarsa_steps'] if 'sarsa_steps' in sarsa_meta else "unknown"
                print(f"Sarsa network was trained with eps={sarsa_eps}, reg={sarsa_reg}, steps={sarsa_steps}")
                if use_action:
                    print(f"objective: {1.0 - action_ratio} * sarsa + {action_ratio} * action_change")
                else:
                    print("Not adding action change objective.")
                self.sarsa_network.load_state_dict(sarsa_ckpt['state_dict'])
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # Random start.
                noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                states = last_states + noise
                if use_action:
                    # Current action at this state.
                    old_action, old_stdev = self.policy_model(last_states)
                    old_stdev /= (old_stdev.mean())
                    old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        # This is the mean action...
                        actions = self.policy_model(states)[0]
                        value = self.sarsa_network(torch.cat((last_states, actions), dim=1)).mean(dim=1)
                        if use_action:
                            action_change = (actions - old_action) / old_stdev
                            # We want to maximize the action change, thus the minus sign.
                            action_change = -(action_change * action_change).mean(dim=1)
                            loss = action_ratio * action_change + (1.0 - action_ratio) * value
                        else:
                            action_change = 0.0
                            loss = value
                        loss.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "advpolicy":
            # Attack using a learned policy network.
            assert self.params.ATTACK_ADVPOLICY_NETWORK is not None
            if not hasattr(self, "attack_policy_network"):
                self.attack_policy_network = self.policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=self.VALUE_CALC == "time",
                                                 activation=self.policy_activation)
                print("Loading adversary policy network", self.params.ATTACK_ADVPOLICY_NETWORK)
                advpolicy_ckpt = torch.load(self.params.ATTACK_ADVPOLICY_NETWORK)
                self.attack_policy_network.load_state_dict(advpolicy_ckpt['adversary_policy_model'])
            # Unlike other attacks we don't need step or eps here.
            # We don't sample and use deterministic adversary policy here.
            perturbations_mean, _ = self.attack_policy_network(last_states)
            # Clamp using tanh.
            perturbed_states = last_states + ch.nn.functional.hardtanh(perturbations_mean) * eps
            """
            adv_perturbation_pds = self.attack_policy_network(last_states)
            next_adv_perturbations = self.attack_policy_network.sample(adv_perturbation_pds)
            perturbed_states = last_states + ch.tanh(next_adv_perturbations) * eps
            """
            return perturbed_states.detach()
        elif self.params.ATTACK_METHOD == "none":
            return last_states
        else:
            raise ValueError(f'Unknown attack method {self.params.ATTACK_METHOD}')



    def apply_ppo_attack(self, last_states):
        eps = self.params.ATTACK_EPS
        if eps == "same":
            eps = self.params.ROBUST_PPO_EPS
        else:
            eps = float(eps)
        if self.params.ATTACK_METHOD == "random":
            # Apply an uniform random noise.
            noise = torch.empty_like(last_states).uniform_(-eps, eps)
            return (last_states + noise).detach()
        elif self.params.ATTACK_METHOD == "advpolicy":
            # Attack using a learned policy network.
            assert self.params.ATTACK_ADVPOLICY_NETWORK is not None
            if not hasattr(self, "attack_policy_network"):
                self.attack_policy_network = self.policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=self.VALUE_CALC == "time",
                                                 activation=self.policy_activation)
                print("Loading adversary policy network", self.params.ATTACK_ADVPOLICY_NETWORK)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                sys.path.append(parent_dir)
                advpolicy_ckpt = torch.load(self.params.ATTACK_ADVPOLICY_NETWORK)
                self.attack_policy_network.load_state_dict(advpolicy_ckpt['adversary_policy_model'])
            # Unlike other attacks we don't need step or eps here.
            # We don't sample and use deterministic adversary policy here.
            perturbations_mean, _ = self.attack_policy_network(last_states)
            # Clamp using tanh.
            # perturbed_states = last_states + ch.nn.functional.hardtanh(perturbations_mean) * eps
            """
            adv_perturbation_pds = self.attack_policy_network(last_states)
            next_adv_perturbations = self.attack_policy_network.sample(adv_perturbation_pds)
            perturbed_states = last_states + ch.tanh(next_adv_perturbations) * eps
            """
            return perturbations_mean  # perturbed_states.detach()
        elif self.params.ATTACK_METHOD == "none":
            return last_states
        else:
            raise ValueError(f'Unknown attack method {self.params.ATTACK_METHOD}')

    @staticmethod
    def agent_from_params(params, store=None):
        '''
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling trajectories, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        - A Trainer object for training a PPO/TRPO agent
        '''
        if params['history_length'] > 0:
            agent_policy = CtsLSTMPolicy
            if params['use_lstm_val']:
                agent_value = ValueLSTMNet
            else:
                agent_value = value_net_with_name(params['value_net_type'])
        else:
            agent_policy = policy_net_with_name(params['policy_net_type'])
            agent_value = value_net_with_name(params['value_net_type'])

        advanced_logging = params['advanced_logging'] and store is not None
        log_every = params['log_every'] if store is not None else 0

        if params['cpu']:
            torch.set_num_threads(1)
        p = Trainer(agent_policy, agent_value, params, store, log_every=log_every,
                    advanced_logging=advanced_logging)

        return p

