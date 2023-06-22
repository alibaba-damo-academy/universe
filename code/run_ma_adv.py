import rllib
import universe
import ray

import time
import psutil
import torch

import numpy as np



# @rllib.basic.system.cpu_memory_profile
def run_one_episode(env, methods):
    t1 = time.time()
    env.reset()
    state = [s.to_tensor().unsqueeze(0) for s in env.state]
    t2 = time.time()
    time_select_action = 0.0
    time_env_step = 0.0

    method = methods[0]
    method_adv = methods[1]

    while True:
        if env.config.render:
            env.render()

        tt1 = time.time()
        state_bg = state[1:]
        if len(state_bg) > 0:
            action, action_adv = ray.get([method.select_action.remote(state[0]), method_adv.select_actions.remote(state_bg)])
        else:
            action = ray.get(method.select_action.remote(state[0]))
            action_adv = torch.zeros((0,action.shape[1]))

        action = torch.cat([action, action_adv], dim=0).numpy()

        tt2 = time.time()
        experience, experience_bg, done, info = env.step(action)
        tt3 = time.time()
        time_select_action += (tt2-tt1)
        time_env_step += (tt3-tt2)

        method.store.remote(experience, index=env.env_index)
        method_adv.store.remote(experience_bg, index=env.env_index)
        state = [s.to_tensor().unsqueeze(0) for s in env.state]
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/select_action', time_select_action, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return




def main():
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rllib.basic.setup_seed(config.seed)
    
    import gallery_ma_adv as gallery
    import models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError



    ################################################################################################
    ##### multi scenario, explicit adv #############################################################
    ################################################################################################

    elif version == 'v1-1':
        writer, env_master, method, method_adv = gallery.ray_sac__multi_scenario__explicit_adv(config, mode)


    elif version == 'v1-2':
        writer, env_master, method, method_adv = gallery.ray_sac__multi_scenario__explicit_adv__small_scale(config, mode)



    ################################################################################################
    ##### multi scenario, implicit adv #############################################################
    ################################################################################################


    elif version == 'v3-1': ### attack driving policy only
        writer, env_master, method, method_adv = gallery.ray_sac__multi_scenario__target_adv(config, mode)



    else:
        raise NotImplementedError


    try:
        env_master.create_tasks((method, method_adv), func=run_one_episode)

        for i_episode in range(10000):
            # total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            # print('update episode i_episode: ', i_episode)
            # ray.get([method.update_parameters_.remote(i_episode, n_iters=sum(total_steps)), method_adv.update_parameters_.remote(i_episode, n_iters=sum(total_steps))])


            total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            num_steps = int(sum(total_steps) /2)
            print('update episode i_episode: ', i_episode)
            if i_episode % 2 == 0:
                ray.get(method.update_parameters_.remote(i_episode, n_iters=num_steps))
            else:
                ray.get(method_adv.update_parameters_.remote(i_episode, n_iters=num_steps))



    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.get(method.close.remote())
        ray.shutdown()





if __name__ == '__main__':
    main()
    
