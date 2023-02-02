import rldev
import universe
import ray


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')

    argparser.add_argument('--num-episodes', default=200, type=int, help='number of episodes.')
    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')
    argparser.add_argument('--invert', action='store_true', help='invert axis (default: False)')
    argparser.add_argument('--render-save', action='store_true', help='save render (default: False)')

    args = argparser.parse_args()
    return args



def run_one_episode(env):
    env.reset()
    while True:
        if env.config.render:
            env.render()

        action = env.action_space.sample()
        experience, done, info = env.step(action)
        if done:
            break    
    return



def main():
    config = rldev.YamlConfig()
    args = generate_args()
    config.update(args)

    ray.init(num_cpus=10, num_gpus=1, include_dashboard=False)

    mode = 'train'
    rldev.setup_seed(config.seed)
    

    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character,
        intersection.config_env__with_character,
        merge.config_env__with_character,
        roundabout.config_env__with_character,
    ])

    from universe import EnvInteractiveMultiAgent as Env
    model_name = Env.__name__
    writer_cls = rldev.PseudoWriter
    writer = rldev.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    env_master = universe.EnvMaster(config, writer, env_cls=Env)

    try:
        env_master.create_envs(func=run_one_episode)
        ray.get([t.run.remote(config.num_episodes) for t in env_master.envs])
    finally:
        ray.shutdown()



if __name__ == '__main__':
    main()
    
