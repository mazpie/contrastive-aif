import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from dense_models import *
from conv_models import *
from world_model import *
import utils

def get_mode(dist, n_samples = 100):
    sample = dist.sample_n(n_samples)
    logprob = dist.log_prob(sample)
    mode_indices = torch.argmax(logprob, dim=0)
    return sample[mode_indices]

def stack_states(rssm_states: list, dim=0):
    return dict(
        mean=torch.stack([state['mean'] for state in rssm_states], dim=dim),
        std=torch.stack([state['std'] for state in rssm_states], dim=dim),
        stoch=torch.stack([state['stoch'] for state in rssm_states], dim=dim),
        deter=torch.stack([state['deter'] for state in rssm_states], dim=dim),
    )


def flatten_state(rssm_state: dict):
    return dict(
        mean=torch.reshape(rssm_state['mean'], [-1, rssm_state['mean'].shape[-1]]),
        std=torch.reshape(rssm_state['std'], [-1, rssm_state['std'].shape[-1]]),
        stoch=torch.reshape(rssm_state['stoch'], [-1, rssm_state['stoch'].shape[-1]]),
        deter=torch.reshape(rssm_state['deter'], [-1, rssm_state['deter'].shape[-1]]),
    )


def detach_state(rssm_state: dict):
    return dict(
        mean=rssm_state['mean'].detach(),
        std=rssm_state['std'].detach(), 
        stoch=rssm_state['stoch'].detach(), 
        deter=rssm_state['deter'].detach(),
    )


def expand_state(rssm_state: dict, n : int):
    return dict(
        mean=rssm_state['mean'].expand(n, *rssm_state['mean'].shape),
        std=rssm_state['std'].expand(n, *rssm_state['std'].shape), 
        stoch=rssm_state['stoch'].expand(n, *rssm_state['stoch'].shape), 
        deter=rssm_state['deter'].expand(n, *rssm_state['deter'].shape),
    )


def get_dist(rssm_state: dict):
    return D.independent.Independent(D.Normal(rssm_state['mean'], rssm_state['std']), 1)


class Agent(nn.Module):
    def __init__(self, 
                    config=None,
                    world_lr=6e-4, 
                    policy_lr=8e-5, 
                    value_lr=8e-5, 
                    device='cuda' if torch.cuda.is_available() else 'cpu', 
                ):
        super().__init__()

        self.config = config

        self.action_dist = config['action_dist'] 
        self.use_rewards = config['use_rewards']
        self.contrastive = config['contrastive']
        
        self._action_size = config['action_size']

        self.wm = WorldModel(config)
        self.world_optim = torch.optim.Adam(utils.get_parameters([self.wm]), lr=world_lr)
            
        self.policy = ActionModel(self._action_size, 230, 200, 3, dist=self.action_dist)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.reinforce = self.action_dist == 'one_hot'

        self.value_model = DenseModel(self.wm._feature_size, (1,), 3, 200)
        if self.reinforce or not self.use_rewards:
            self.value_target = DenseModel(self.wm._feature_size, (1,), 3, 200)
        else:
            self.value_target = self.value_model
        self.value_optim = torch.optim.Adam(self.value_model.parameters(), lr=value_lr)

        self.grad_clip = 100.
        self.gamma = config['discount_gamma'] 
        self.device = device

        self.add_actor_entropy = config.get('actor_entropy', False)
        self.entropy_temperature = config.get('entropy_temperature', 1e-4)
        
        # Default for the moment
        self.use_rms = False
        self.rew_rms = utils.RunningMeanStd()
        self.ambiguity_rms = utils.RunningMeanStd()
        self.ambiguity_beta = 1e-3
        ##

        self.to(device)

    def update_target_network(self, tau, network, target_network):
        # Softly Update Target 
        target_value_params = target_network.named_parameters()
        value_params = network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
        
        target_network.load_state_dict(value_state_dict)

    def train_world(self, path_obs, path_act, path_rew, preferred_obs, path_done=None, update_target=False, get_reconstruction=False):
        loss_dict = dict()
        batch_t, batch_b, img_shape = path_obs.shape[0], path_obs.shape[1], path_obs.shape[2:]
        batch_t -= 1

        init_obs = path_obs[0]
        next_obs = path_obs[1:]

        obs_embed = self.wm.obs_encoder(path_obs) 
        init_embed = obs_embed[0]
        next_embed = obs_embed[1:]

        prev_state = self.wm.prior.initial_state(batch_size=batch_b, device=self.device)

        prior, post = self.rollout_posterior(batch_t, next_embed, path_act, prev_state) 
        feat = torch.cat((post['stoch'], post['deter']), dim=-1)

        if self.use_rewards:
            reward_pred = self.wm.rew_model(feat)
            reward_loss = -torch.mean(reward_pred.log_prob(path_rew))
        else:
            reward_loss = torch.zeros(1).to(self.device)

        if self.contrastive:
            W_c = post['stoch'].reshape(batch_t * batch_b, -1) # N
            reshaped_obs = path_obs[1:].reshape(batch_b*batch_t, *img_shape)

            W_c = self.wm.z_encoder(feat).reshape(batch_t * batch_b, -1)
            mean_z = self.wm.w_contrastive(next_embed).reshape(batch_t * batch_b, -1)
            sim_matrix = torch.mm(W_c, mean_z.T) 
            labels = torch.Tensor(list(range(batch_b * batch_t))).long().to(self.device)
            image_loss = F.cross_entropy(sim_matrix, labels, reduction='mean') 
        else:
            image_pred = self.wm.obs_decoder(feat)
            image_loss = -torch.mean(image_pred.log_prob(next_obs))

        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
            
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        kl_loss = torch.max(div, div.new_full(div.size(), self.wm.free_nats))

        model_loss = image_loss + reward_loss + self.wm.kl_scale * kl_loss 

        self.world_optim.zero_grad()
        model_loss.backward()

        grad_norm_world = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.wm]), self.grad_clip)

        loss_dict['world_grad_norm'] = grad_norm_world
        
        self.world_optim.step()
        
        loss_dict = dict(reconstruction_loss=image_loss.item(), kl_loss=kl_loss, reward_loss=reward_loss.item(), **loss_dict)

        if get_reconstruction and (not self.contrastive):
            with torch.no_grad():
                vb = 16 # video batch size
                
                true_steps = 5
                ground_truth = (path_obs[1:, :vb] + 0.5).cpu() 
                recon_truth = next_obs.cpu()[:, :vb] + 0.5
                recon = image_pred.mean.detach()[:, :vb]

                init = {k: v[true_steps-1, :vb] for k,v in post.items()} 
                rec_prior = self.rollout_prior(batch_t - true_steps, path_act[true_steps:, :vb], init)
                rec_feat = torch.cat((rec_prior['stoch'], rec_prior['deter']), dim=-1)
                future_pred = self.wm.obs_decoder(rec_feat).mean

                model = torch.clamp(torch.cat([recon[:true_steps], future_pred], dim=0) + 0.5, 0, 1).cpu()
                model_post = torch.clamp(recon + 0.5, 0, 1).cpu()

                error = (model - recon_truth + 1) / 2
                post_prior_div = (model_post - model + 1) / 2

                reconstruction_dict = dict(truth=ground_truth, rencostructed_truth=recon_truth, prior_predictions=model, post_predictions=model_post, prior_truth_diff=error, post_prior_diff=post_prior_div)
        else:
            reconstruction_dict = dict()

        return post, loss_dict, reconstruction_dict

    def train_value(self, state_features, lambda_returns, discount_arr):
        value_pred = self.value_model(state_features.detach())
        value_pred = D.independent.Independent(D.Normal(value_pred.mean[:-1], 1), 1) 

        value_loss = -torch.mean(discount_arr * value_pred.log_prob(lambda_returns.detach()) )

        self.value_optim.zero_grad()
        value_loss.backward()
        grad_norm_value = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.value_model]), self.grad_clip)
        
        self.value_optim.step()

        return value_loss.item(), grad_norm_value

    def train_policy_value(self, steps, policy, states, preferred_obs, obs_batch=None):
        with utils.FreezeParameters([self.wm, self.value_model]):
            states = detach_state(flatten_state(states))
            
            list_prior_states, act_logprobs, actions, act_entropies = self.rollout_policy(steps, policy, states)

            prior_states =  stack_states(list_prior_states, dim=0)

            all_prior_feat = torch.cat((prior_states['stoch'], prior_states['deter']), dim=-1)

            if self.use_rewards:
                future_rew_pred = self.wm.rew_model(all_prior_feat)
                free_energy, fe_dict = self.compute_free_energy(future_rew_pred.mean, preferred_obs, prior_states, actions)
            else:
                free_energy, fe_dict = self.compute_free_energy(None, preferred_obs, prior_states, actions, obs_batch=obs_batch)

            loss_dict = dict(**fe_dict)

            future_value_pred = self.value_target(all_prior_feat).mean

            if not self.use_rewards:
                if self.use_rms:
                    self.rew_rms.update(free_energy.detach().cpu().view(-1, 1).numpy())
                    free_energy = free_energy / np.sqrt(self.rew_rms.var.item() + 1e-8)

            loss_dict['free_energy'] = free_energy.detach().mean().item()
            loss_dict['action_entropy'] = act_entropies.mean().detach().cpu().item()

            discount = torch.ones_like(future_value_pred) * self.gamma

            expected_free_energy = utils.lambda_dreamer_values(free_energy[:,:,None], future_value_pred, gamma=discount)
            loss_dict['expected_free_energy'] = expected_free_energy.detach().cpu().mean().item()

            discount_arr = torch.cat( [discount[:1] / self.gamma, discount[:-1]], dim=0).detach() 
            discount_arr = torch.cumprod(discount_arr, dim=0).squeeze(-1)

            if self.reinforce:
                future_value_pred = self.value_model(all_prior_feat).mean
                advantages = (expected_free_energy - future_value_pred).detach().squeeze(-1)
                loss = torch.mean(discount_arr[:-1] * advantages[:-1]  * act_logprobs)
            else:
                loss = torch.mean(discount_arr[:-1] * expected_free_energy[:-1].squeeze(-1))

            if self.add_actor_entropy:
                loss = loss - torch.mean(discount_arr[:-1] * act_entropies * self.entropy_temperature)

            self.policy_optim.zero_grad()
            
            loss.backward()
            
            grad_norm_actor = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.policy]), self.grad_clip)
            loss_dict['policy_grad_norm'] = grad_norm_actor

            self.policy_optim.step()

        with utils.FreezeParameters([self.wm, self.policy]):
            value_loss_item, value_grad_norm = self.train_value(all_prior_feat, expected_free_energy[:-1].detach(), discount_arr[:-1])            
            loss_dict['value_logprob_loss'] = value_loss_item
            loss_dict['value_grad_norm'] = value_grad_norm

        return loss_dict 

    def compute_free_energy(self, rew, preferred_obs, prior_states=None, actions=None, obs_batch=None):
        free_energy_dict = dict()
        free_energy = torch.zeros(1,1)
        
        if len(prior_states['stoch'].shape) == 2:
            prior_states = expand_state(prior_states,1) 
        batch_t, batch_b, state_dim = prior_states['stoch'].shape[0], prior_states['stoch'].shape[1], prior_states['stoch'].shape[2]

        if self.use_rewards:
            free_energy = -rew.squeeze(-1)
            preferences = free_energy
        else:
            # Contrastive AIF
            if self.contrastive:
                feat = torch.cat((prior_states['stoch'], prior_states['deter']), dim=-1)
                pref_embed = self.wm.obs_encoder(preferred_obs).view(1, self.wm.obs_encoder.embed_size)

                W_c = self.wm.z_encoder(feat).reshape(batch_t * batch_b, -1)     
                pref_z = self.wm.w_contrastive(pref_embed).reshape(1, 200)
                        
                pos_loss = torch.mm(W_c, pref_z.T).view(batch_t, batch_b) / W_c.shape[-1]
                free_energy = -pos_loss
                free_energy_dict['-pos_loss'] = pos_loss.detach().mean().item()

                # Only to allow computation along the episode
                if obs_batch is not None:
                    next_embed = self.wm.obs_encoder(obs_batch).view(-1, self.wm.obs_encoder.embed_size)

                    mean_z = self.wm.w_contrastive(next_embed).reshape(-1, 200)

                    mean_z = torch.cat([pref_z, mean_z], dim=0)
                    sim_matrix = torch.mm(W_c, mean_z.T)  / W_c.shape[-1]
                    neg_loss = torch.logsumexp(sim_matrix, dim=1).view(batch_t, batch_b) - np.log(mean_z.shape[0])
                    free_energy = -pos_loss + neg_loss
                    free_energy_dict['neg_loss'] = neg_loss.detach().mean().item()
            # Likelihood AIF
            else:
                feat = torch.cat((prior_states['stoch'], prior_states['deter']), dim=-1)

                image_pred = self.wm.obs_decoder(feat)
                predicted_obs = image_pred.mean
                preferred_obs_dist = D.Independent(D.Laplace(preferred_obs, 1.), 3)

                logprob_preferences = preferred_obs_dist.log_prob(predicted_obs) / np.prod(preferred_obs.shape)
                free_energy = - logprob_preferences
                free_energy_dict['-logprob_preferences'] = (-logprob_preferences).detach().mean().item()

                init_states = flatten_state(prior_states)
                _, posterior_states = self.wm.posterior(obs_embed=self.wm.obs_encoder(preferred_obs).expand(batch_b*batch_t, self.wm.obs_encoder.embed_size), prev_action=None, prev_state=init_states, is_init=True)
                prior_dist = get_dist(init_states)
                post_dist = get_dist(posterior_states)
                epistemic_term = D.kl_divergence(post_dist, prior_dist).reshape(*logprob_preferences.shape) / init_states['stoch'].shape[-1]

                self.ambiguity_rms.update(epistemic_term.detach().cpu().view(-1, 1).numpy())
                epistemic_term = epistemic_term / np.sqrt(self.ambiguity_rms.var.item() + 1e-8)

                free_energy = - logprob_preferences - self.ambiguity_beta * epistemic_term
                free_energy_dict['-epistemic_term'] = (-epistemic_term).detach().mean().item()

        return free_energy, free_energy_dict


    def rollout_policy(self, steps, policy, prev_state):
        priors = [prev_state]
        act_logprobs = []
        actions = []
        act_entropies = []
        state = prev_state
        for t in range(steps):
            # Act
            feat = torch.cat((state['stoch'], state['deter']), dim=-1)
            feat = feat.detach() 
            act_dist = policy(feat)

            if self.reinforce:
                act = act_dist.sample()
            else:
                act = act_dist.rsample() 
            
            actions.append(act)
            act_entropies.append(act_dist.entropy())
            act_logprobs.append(act_dist.log_prob(act))

            # Imagine
            state = self.wm.prior(actions[t], state)
            priors.append(state)
        
        actions = torch.stack(actions, dim=0) # shape (T, B, action_dim)
        act_logprobs = torch.stack(act_logprobs, dim=0) # shape (T, B, action_dim)
        act_entropies = torch.stack(act_entropies, dim=0) # shape (T, B, action_dim)

        return priors, act_logprobs, actions, act_entropies

    def rollout_policies(self, steps, policies, prev_state, take_mean_action=False):
        priors = []
        actions = []
        state = prev_state
        for t in range(steps):
            # Act
            feat = torch.cat((state['stoch'], state['deter']), dim=-1)
            act = torch.stack([ p(f).sample() for f, p in zip(feat, policies)], dim=0)
            actions.append(act)
            # Imagine
            state = self.wm.prior(actions[t], state)
            priors.append(state)
        
        all_prior_states =  stack_states(priors, dim=0)
        actions = torch.stack(actions, dim=1) # shape: (B,T,action_dim)
        return all_prior_states, actions

    def step(self, image_obs, rew, act, prev_state):
        with torch.no_grad():
            image_embed = self.wm.obs_encoder(image_obs)
            if prev_state is None:
                prev_state = self.wm.posterior.initial_state(batch_size=1, device=self.device)
            _, post = self.rollout_posterior(1, image_embed, act, prev_state)
        return flatten_state(post)

    def eval_obs(self, image_obs, rew, preferred_obs, prev_state):
        with torch.no_grad(): 
            if self.use_rewards:
                preferences = self.compute_free_energy(rew, preferred_obs, prior_states=prev_state)[0]
            else:
                preferences = self.compute_free_energy(None, preferred_obs, prior_states=prev_state)[0]
        return preferences

    def policy_distribution(self, steps, policies, preferred_obs, prev_state=None, eval_mode=False):
        with torch.no_grad():
            n_policies = len(policies)
            if prev_state is None:
                prev_state = self.wm.posterior.initial_state(batch_size=n_policies, device=self.device)

            prev_state = flatten_state(prev_state)
            all_prior_states, actions = self.rollout_policies(steps, policies, prev_state, take_mean_action=eval_mode)
            all_prior_feat = torch.cat((all_prior_states['stoch'], all_prior_states['deter']), dim=-1)

            if self.use_rewards:
                future_rew_pred = self.wm.rew_model(all_prior_feat)
                preference_loss = torch.mean(self.compute_free_energy(future_rew_pred.mean, preferred_obs, prior_states=all_prior_states)[0], dim=0)
            else:
                preference_loss = torch.mean(self.compute_free_energy(None, preferred_obs, prior_states=all_prior_states)[0], dim=0)

            free_energy = preference_loss
            policy_logits = F.softmax(-free_energy.detach(), dim=0)
            
            policy_distr = D.Categorical(policy_logits)
            expected_loss = torch.sum(free_energy.detach() * policy_logits)
            return policy_distr, actions.detach(), dict(policy_expected_loss=expected_loss.detach().cpu().item())

    def rollout_posterior(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor, prev_state: dict):
        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, posterior_state = self.wm.posterior(obs_embed[t], action[t], prev_state, transition_model=self.wm.prior)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post

    def rollout_prior(self, steps: int, action: torch.Tensor, prev_state: dict):
        priors = []
        state = prev_state
        for t in range(steps):
            state = self.wm.prior(action[t], state)
            priors.append(state)
        return stack_states(priors, dim=0)
