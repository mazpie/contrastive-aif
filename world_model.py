import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from dense_models import *
from conv_models import *

class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        ConvModule = ObservationEncoder
        DeconvModule = ObservationDecoder

        self.free_nats = 3.
        self.kl_scale = 1.

        self.obs_encoder = ConvModule()

        if config['contrastive']: 
            self.z_encoder = nn.Sequential(   
                                    nn.Linear(230, 200),
                                    nn.ELU(),
                                    nn.Linear(200, 200),
                                    nn.Tanh(),
                                )     

            self.w_contrastive = nn.Sequential(   
                                    nn.Linear(self.obs_encoder.embed_size, 400),
                                    nn.ELU(),
                                    nn.Linear(400, 200),
                                    nn.Tanh(),
                                )
            obs_modules = [self.z_encoder, self.w_contrastive]
        else:
            self.obs_decoder = DeconvModule()
            obs_modules = [self.obs_decoder]

        self._embed_size = self.obs_encoder.embed_size
        
        self.prior = TransitionModel(config['action_size'])
        self.posterior = RepresentationModel(self._embed_size, config['action_size'])
        
        self._hidden_size = 200
        self._deter_size = 200
        self._stoch_size = 30
        self._feature_size = self._deter_size + self._stoch_size

        if config['use_rewards']:
            self.rew_model = DenseModel(self._feature_size, (1,), 3, 300)

class TransitionModel(nn.Module):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200, activation=nn.ELU,
                 distribution=D.Normal):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._rnn_input_model = self._build_rnn_input_model()
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution

    def _build_rnn_input_model(self):
        rnn_input_model = [nn.Linear(self._action_size + self._stoch_size, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return dict(
            mean=torch.zeros(batch_size, self._stoch_size, **kwargs),
            std=torch.zeros(batch_size, self._stoch_size, **kwargs),
            stoch=torch.zeros(batch_size, self._stoch_size, **kwargs),
            deter=torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, prev_action: torch.Tensor, prev_state: dict):
        rnn_input = self._rnn_input_model(torch.cat([prev_action, prev_state['stoch']], dim=-1))
        deter_state = self._cell(rnn_input, prev_state['deter'])
        mean, std = torch.chunk(self._stochastic_prior_model(deter_state), 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = D.Independent(self._dist(mean, std), 1)
        stoch_state = dist.rsample() 
        return dict(mean=mean, std=std, stoch=stoch_state, deter=deter_state)

class RepresentationModel(nn.Module):
    def __init__(self, obs_embed_size, action_size, stochastic_size=30,
                 deterministic_size=200, hidden_size=200, activation=nn.ELU, distribution=D.Normal):
        super().__init__()
        self._obs_embed_size = obs_embed_size
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._stochastic_posterior_model = self._build_stochastic_model()

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size + self._stoch_size + self._obs_embed_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return dict(
            mean=torch.zeros(batch_size, self._stoch_size, **kwargs),
            std=torch.zeros(batch_size, self._stoch_size, **kwargs),
            stoch=torch.zeros(batch_size, self._stoch_size, **kwargs),
            deter=torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, obs_embed: torch.Tensor, prev_action: torch.Tensor, prev_state: dict, is_init=False, transition_model=None):
        if is_init:
            prior_state = prev_state
        else:
            prior_state = transition_model(prev_action, prev_state)
        x = torch.cat([prior_state['stoch'], prior_state['deter'], obs_embed], dim=-1)
        mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = D.Independent(self._dist(mean, std), 1)
        stoch_state = dist.rsample()
        posterior_state = dict(mean=mean, std=std, stoch=stoch_state, deter=prior_state['deter'])
        return prior_state, posterior_state