
import ray


class MultiAgent(object):
    @staticmethod
    def v0(env, method):
        env.reset()
        state = [s.to_tensor().unsqueeze(0) for s in env.state]
        while True:
            if env.config.render:
                env.render()

            action = ray.get(method.select_actions.remote(state)).cpu().numpy()
            experience, done, info = env.step(action)

            method.store.remote(experience, index=env.env_index)
            state = [s.to_tensor().unsqueeze(0) for s in env.state]
            if done:
                break
        return




class SingleAgent(object):
    @staticmethod
    def v0(env, method):
        env.reset()
        state = env.state[0].to_tensor().unsqueeze(0)
        while True:
            if env.config.render:
                env.render()

            action = ray.get(method.select_action.remote(state)).cpu().numpy()
            experience, done, info = env.step(action)
            method.store.remote(experience, index=env.env_index)

            state = env.state[0].to_tensor().unsqueeze(0)
            if done:
                break
        return


    @staticmethod
    def v1(env, method):
        env.reset()
        state = env.state[0].to_tensor().unsqueeze(0)
        while True:
            if env.config.render:
                env.render()

            action = method.select_action(state).cpu().numpy()
            experience, done, info = env.step(action)
            method.store(experience, index=env.env_index)

            state = env.state[0].to_tensor().unsqueeze(0)
            if done:
                break
        return
    