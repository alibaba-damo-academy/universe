import rllib
import ray
import psutil, torch



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
    
    import gallery_ma as gallery
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError


    elif version == 'social_comm':
        runner, method = gallery.isac__social_comm(config, mode)


    else:
        raise NotImplementedError


    try:
        runner.create_tasks(method)
        for i_episode in range(10000):
            runner.execute(i_episode, method)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.get(method.close.remote())
        ray.shutdown()





if __name__ == '__main__':
    main()
    
