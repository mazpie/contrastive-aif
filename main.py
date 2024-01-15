import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
import argparse
import os
import warnings
import ruamel.yaml as yaml
import pathlib
import sys
from pathlib import Path
import datetime
import json

import utils
import wrappers
from agents import Agent

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
# os.environ['MUJOCO_GL'] = 'egl'

def visualize_current_obs(obs):
    import matplotlib.pyplot as plt
    plt.imshow(obs.cpu().permute(1,2,0) + 0.5)
    plt.show()

@utils.retry
def save_model(model, save_dir):
    torch.save(model, save_dir)

def collect_random_episode(env, model, preferred_obs, episode_store, returns, free_energies, config):
    with torch.no_grad():
        device = config['device']
        episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])
        timestep = env.reset()
        obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
        episode['obs'].append(obs.cpu())
        prev_state = None

        done = False
        cur_return = 0
        cur_free_energy = 0
        while not done:
            action = env.action_space.sample()

            timestep, rew, done, info = env.step(action)
            obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
            
            rew_tensor = torch.Tensor([rew]).to(device)
            # Minigrid only: to simplify reward prediction, as the agent has no knowledge of time
            if 'minigrid' in config['suite'] and rew_tensor > 0:
                rew_tensor = torch.ones_like(rew_tensor)
            #
            done_tensor = torch.Tensor([done]).to(device)
            current_obs = obs.expand(1, 1, *obs.shape)
            prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1,1,1).to(device), torch.from_numpy(action).to(device).view(1, 1, *action.shape).to(device), prev_state)

            free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)
            episode['obs'].append(obs.cpu())
            episode['act'].append(torch.from_numpy(action))
            episode['rew'].append(rew_tensor.cpu())
            episode['done'].append(done_tensor.cpu())
            episode['free_energy'].append(free_energy.cpu())

            cur_return += rew
            cur_free_energy += free_energy.cpu().numpy()
        
        episode_store.add(np.stack(episode['obs'], axis=0), np.stack(episode['act'], axis=0), np.stack(episode['rew'], axis=0), np.stack(episode['free_energy'], axis=0), np.stack(episode['done'], axis=0))
        returns.append(cur_return)
        free_energies.append(cur_free_energy)


def collect_eval_episode(env, model, preferred_obs, config, policy_lookahead=1):
    with torch.no_grad():
        device = config['device']
        # Reset
        timestep = env.reset()
        obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
        prev_state = None
        cur_return = 0
        cur_free_energy = 0
        policy_dict = dict()
        done = False
        while not done:
            policy_distr, policy_actions, future_loss_dict = model.policy_distribution(policy_lookahead, [model.policy], preferred_obs, prev_state, eval_mode=True)

            for k in future_loss_dict:
                if k in policy_dict:
                    policy_dict[k] += future_loss_dict[k]
                else:
                    policy_dict[k] = future_loss_dict[k]
        
            # Action
            for policy_step in range(policy_lookahead):
                action_index = policy_distr.sample()
                action = policy_actions[action_index][policy_step].cpu()
                
                timestep, rew, done, info = env.step(action.numpy())
                obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
                
                rew_tensor = torch.Tensor([rew]).to(device)
                # Minigrid only: to simplify reward prediction, as the agent has no knowledge of time
                if 'minigrid' in config['suite'] and rew_tensor > 0:
                    rew_tensor = torch.ones_like(rew_tensor)
                #
                done_tensor = torch.Tensor([done]).to(device)
                current_obs = obs.expand(1, 1, *obs.shape)
                prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1,1,1).to(device), action.view(1, 1, *action.shape).to(device), prev_state)
                
                free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)

                cur_return += rew
                cur_free_energy += free_energy.cpu().numpy()
                
                if done:
                    break
    return cur_return, cur_free_energy 


def main(config):   
    # Setup logs
    if config['seed'] is not None:
        seed = int(config['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
    seed_str = str(datetime.datetime.now().timestamp()) if config['seed'] is None else 'seed_' + str(seed)
    logdir = Path(config['logdir']) /  config['suite'] / config['task'] / '_'.join(config['config']) / config['algo'] / seed_str

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir=str(logdir))
    with open(str(logdir/'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)

    # Options
    device = 'cuda:0' if torch.cuda.is_available() and not config['disable_cuda'] else 'cpu'
    config['device'] = device

    # Create env
    env = wrappers.make_env(suite=config['suite'], task_name=config['task'], grid_size=config['grid_size'])
    config['action_size'] = env.action_space.shape[0]
    action_repeat = env._action_repeat

    # Setup model
    model = Agent(device=device, config=config)
    
    if config['suite'] == 'minigrid_pixels':
        if config['task'] == 'empty':
            preferred_obs = wrappers.get_scaled_obs(dict(image=torch.load(f'preferred_states/empty_goal_dir_0.pt')*255), device, is_minigrid=False).permute(2,0,1) 
            preferred_obs = torch.nn.functional.pad(preferred_obs, (0, 1, 0, 1))
        else:
            raise NotImplementedError('No preferred state defined')
    elif config['use_rewards']:
        preferred_obs = None 
    else:
        preferred_obs = wrappers.get_scaled_obs(dict(image=torch.load(f'preferred_states/{config["task"]}.pt')), device, is_minigrid='minigrid' in config['suite']) 

    policy_lookahead = 1
    expl_amount = config['expl_amount'] 

    # Setup training
    total_steps = int(config['total_steps'])
    max_episodes = config['max_episodes'] if config['max_episodes'] is not None else int(sys.maxsize)
    episode_store = utils.EpisodeStore()
    balance_ends = config['action_dist'] == 'one_hot'

    current_step = 0
    tot_episodes = 0
    century = 0

    # Populate episode store
    random_init_episodes = config['random_init_episodes']
    print(f"Collecting {random_init_episodes} episodes for init...")
    init_returns = []
    init_free_energies = []
    while len(init_returns) < random_init_episodes:
        collect_random_episode(env, model, preferred_obs, episode_store, init_returns, init_free_energies, config)
    print("Random collection completed...", ' Return: ', np.round(np.mean(init_returns), 2))
    writer.add_scalar('environment/return', np.mean(init_returns), global_step=current_step)
    writer.add_scalar('environment/free_energy', np.mean(init_free_energies), global_step=current_step)

    # Reset
    episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])
    timestep = env.reset()
    if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:
        env.render()
    obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
    episode['obs'].append(obs.cpu())
    prev_state = None
    cur_return = 0
    policy_dict = dict()
    done = True

    while current_step < total_steps:
        if tot_episodes >= max_episodes:
            print('Maximum Episodes reached!')
            break

        if done and episode_store.n_episodes > 0:
            print(f'E: {tot_episodes} | Updating model and policies...')
            n_epochs = config['n_epochs']  
            n_paths = config['n_paths']    
            n_steps = config['n_steps']    
            horizon = config['horizon']    
            main_loss_dict = dict()
            main_policy_loss_dict = dict()
            for i in range(n_epochs):
                log_time = i == n_epochs - 1
                log_images =  (config['recon_every'] > 0) and (tot_episodes % config['recon_every'] == 0) and log_time
                path_obs, path_act, path_rew, path_done = episode_store.sample_paths(n_paths, n_steps, balance_ends=balance_ends) 
                
                update_target = i + 1 % 100 == 0 
                states, loss_dict, reconstruction_dict = model.train_world(path_obs, path_act, path_rew, preferred_obs,  path_done=path_done,
                                        update_target=update_target, get_reconstruction=log_images)
                
                policy_loss_dict = model.train_policy_value(horizon, model.policy, states, preferred_obs, obs_batch=path_obs)
                
                for k in loss_dict:
                    if k in main_loss_dict:
                        main_loss_dict[k] += loss_dict[k] / n_epochs
                    else:
                        main_loss_dict[k] = loss_dict[k] / n_epochs

                for k in policy_loss_dict:
                    if k in main_policy_loss_dict:
                        main_policy_loss_dict[k] += policy_loss_dict[k] / n_epochs
                    else:
                        main_policy_loss_dict[k] = policy_loss_dict[k] / n_epochs
            # Logging
            for k,v in main_loss_dict.items():
                writer.add_scalar('world_model/' + k, v, global_step=current_step)
            if len(reconstruction_dict) > 0:
                print('Logging videos')
            for k, v in reconstruction_dict.items():

                if config['suite'] == 'minigrid':
                    new_tensor = []
                    for k_group, v_group in enumerate(v):
                        new_group = []
                        for k_frame, v_frame in enumerate(v[k_group]):
                            new_group.append(env._env._env.get_obs_render(
                                    torch.clamp(v_frame, 0, 5).permute(1,2,0).round().int().numpy(),
                                    tile_size=8
                            ))
                        new_tensor.append(new_group)
                    v = torch.tensor(new_tensor).permute(0,1,4,2,3)
                
                writer.add_video(k, v.permute(1,0,2,3,4), global_step=current_step, fps=15)
            for k, v in main_policy_loss_dict.items():
                writer.add_scalar('policy/' + k, v, global_step=current_step)
            if model.reinforce or not config['use_rewards']:
                print('Updating Value Model Target...')
                model.update_target_network(1., model.value_model, model.value_target)
            # Save model
            if config['save_every'] > 0 and tot_episodes % config['save_every'] == 0:
                save_model(model, str(logdir / f'world_model.pt'))
            
        done = False

        while not done:
            # Policy Selection
            if current_step // 100 > century:
                print('Step: ', current_step)
                century += 1

            policy_distr, policy_actions, future_loss_dict = model.policy_distribution(policy_lookahead, [model.policy], preferred_obs, prev_state)

            for k in future_loss_dict:
                if k in policy_dict:
                    policy_dict[k] += future_loss_dict[k]
                else:
                    policy_dict[k] = future_loss_dict[k]
        
            # Action
            for policy_step in range(policy_lookahead):
                action_index = policy_distr.sample()
                action = policy_actions[action_index][policy_step].cpu()
                if config['action_dist'] == 'one_hot':
                    if np.random.rand() < expl_amount:
                        action = torch.from_numpy(env.action_space.sample()).float()
                    expl_amount = max(config['expl_amount'] * (config['expl_steps'] * action_repeat - current_step) / (config['expl_steps'] * action_repeat), 0)
                else:
                    act_noise = torch.randn(*action.shape, device=action.device) * expl_amount
                    action = torch.clamp(action + act_noise, -1, 1)
                
                timestep, rew, done, info = env.step(action.numpy())
                if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:
                    env.render()
                obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
                
                rew_tensor = torch.Tensor([rew]).to(device)
                # Minigrid only: to simplify reward prediction, as the agent has no knowledge of time
                if 'minigrid' in config['suite'] and rew_tensor > 0:
                    rew_tensor = torch.ones_like(rew_tensor)
                #
                done_tensor = torch.Tensor([done]).to(device)
                current_obs = obs.expand(1, 1, *obs.shape)
                prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1,1,1).to(device), action.view(1, 1, *action.shape).to(device), prev_state)
                
                free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)
                episode['obs'].append(obs.cpu())
                episode['act'].append(action.cpu())
                episode['rew'].append(rew_tensor.cpu())
                episode['done'].append(done_tensor.cpu())
                episode['free_energy'].append(free_energy.cpu())

                current_step += 1  * action_repeat
                cur_return += rew
                
                if done:
                    tot_episodes += 1
                    episode_store.add(np.stack(episode['obs'], axis=0), np.stack(episode['act'], axis=0), np.stack(episode['rew'], axis=0), np.stack(episode['free_energy'], axis=0), np.stack(episode['done'], axis=0))
                    # print('Step: ', current_step, ' Return: ', np.round(cur_return, 2), 'Expected Free Energy: ', np.round(policy_dict['policy_expected_loss'],2) )
                    writer.add_scalar('environment/return', cur_return, global_step=current_step)
                    writer.add_scalar('environment/return_over_episodes', cur_return, global_step=tot_episodes)
                    writer.add_scalar('environment/free_energy', np.sum(episode['free_energy']), global_step=current_step)
                    for k in policy_dict:
                        writer.add_scalar('environment/' + k, policy_dict[k] , global_step=current_step)
                    
                    # Record and/or eval
                    if config['record_every'] > 0 and tot_episodes % config['record_every'] == 0:
                        writer.add_video('environment/train_episode', np.expand_dims(np.stack(episode['obs'], axis=0) + 0.5, 0), global_step=tot_episodes, fps=15)

                    episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])
                    timestep = env.reset()
                    if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:
                        env.render()
                    
                    obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
                    episode['obs'].append(obs.cpu())

                    prev_state = None
                    cur_return = 0
                    policy_dict = dict()
                    # This breaks the policy_step cycle
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--suite', help='suite name', default='dmc')
    parser.add_argument('--task', help='task name', default='cheetah_run')
    parser.add_argument('--algo', help='algo name', default='dreamer')
    parser.add_argument('--config', nargs='+', help='configuration name', default=['base'])
    
    parser.add_argument('--disable-cuda', help='disable cuda', action='store_true', default=False)
    parser.add_argument('--seed', help='set random seed', default=None, type=int)
    
    parser.add_argument('--save-every', help='save model', default=0, type=int)
    parser.add_argument('--render-every', help='render agent', default=0, type=int)
    parser.add_argument('--record-every', help='record training episodes', default=100, type=int)
    parser.add_argument('--recon-every', help='log reconstructions as videos', default=0, type=int)
    
    args = parser.parse_args()
    configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    conf = dict()
    for k in ['base', *args.config]:
        conf = dict(conf, **configs[k])
    conf = dict(conf, **configs['algos'][args.algo])
    for k in args.__dict__:
        conf[k] = args.__dict__[k]
    print(conf)
    main(conf)
    