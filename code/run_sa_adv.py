import rllib
import universe
import ray

import time
import psutil
import torch



def run_one_episode(env, method):
    t1 = time.time()
    env.reset()
    state = env.state[0].to_tensor().unsqueeze(0)
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        state_id = ray.put(state)
        action, action_adv = ray.get([method.select_action.remote(state_id), method.select_action_adv.remote(state_id)])
        action = action.cpu().numpy()
        action_adv = action_adv.cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action, action_adv)
        tt2 = time.time()
        time_env_step += (tt2-tt1)

        experience_id = ray.put(experience)
        method.store.remote(experience_id, index=env.env_index)
        state = env.state[0].to_tensor().unsqueeze(0)
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
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
    
    import gallery_sa_adv as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError




    elif version == 'v5-1':
        writer, env_master, method = gallery.ray_sac__multi_scenario__adaptive_adv_background(config, mode)


        # 验证时： 15600



    else:
        raise NotImplementedError
    
    try:
        env_master.create_tasks(method, func=run_one_episode)

        for i_episode in range(10000):
            total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            num_steps = int(sum(total_steps) /2)
            print('update episode i_episode: ', i_episode)
            if i_episode % 2 == 0:
                ray.get(method.update_parameters_.remote(i_episode, n_iters=num_steps))
            else:
                ray.get(method.update_parameters_adv_.remote(i_episode, n_iters=num_steps))

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.get(method.close.remote())
        ray.shutdown()





if __name__ == '__main__':
    main()
    
