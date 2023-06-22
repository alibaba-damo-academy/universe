import rllib

import numpy as np
import torch



class ReplayBufferCls(rllib.buffer.ReplayBuffer):
    def _batch_stack(self, batch):  ### todo: simplify
        result = rllib.buffer.stack_data(batch)

        state, next_state = result.state, result.next_state

        self.pad_state(state)
        self.pad_state(next_state)
        state.pop('agent_masks')
        state.pop('vehicle_masks')
        next_state.pop('agent_masks')
        next_state.pop('vehicle_masks')
        result.update(state=state)
        result.update(next_state=next_state)

        result = result.cat(dim=0)
        result.vi.unsqueeze_(1)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result


    def pad_state(self, state: rllib.basic.Data):
        pad_state_with_characters(state)


def pad_state_with_characters(state: rllib.basic.Data):
    def pad(element, pad_value):
        sizes = torch.tensor([list(e.shape) for e in element])
        max_sizes = torch.Size(sizes.max(dim=0).values)
        return [pad_data(e, max_sizes, pad_value) for e in element]

    obs = pad(state.obs, pad_value=np.inf)
    obs_mask = pad(state.obs_mask, pad_value=0)
    obs_character = pad(state.obs_character, pad_value=np.inf)

    lane = pad(state.lane, pad_value=np.inf)
    lane_mask = pad(state.lane_mask, pad_value=0)

    bound = pad(state.bound, pad_value=np.inf)
    bound_mask = pad(state.bound_mask, pad_value=0)

    state.update(obs=obs, obs_mask=obs_mask, obs_character=obs_character, lane=lane, lane_mask=lane_mask, bound=bound, bound_mask=bound_mask)
    return




def pad_data(data: torch.Tensor, pad_size: torch.Size, pad_value=np.inf):
    """
    Args:
        data, pad_size: torch.Size([batch_size, dim_elements, dim_points, dim_features])
    """
    res = torch.full(pad_size, pad_value, dtype=data.dtype, device=data.device)

    if len(pad_size) == 2:
        batch_size, dim_elements = data.shape
        res[:batch_size, :dim_elements] = data
    elif len(pad_size) == 3:
        batch_size, dim_elements, dim_points = data.shape
        res[:batch_size, :dim_elements, :dim_points] = data
    elif len(pad_size) == 4:
        batch_size, dim_elements, dim_points, dim_features = data.shape
        res[:batch_size, :dim_elements, :dim_points, :dim_features] = data
    else:
        raise NotImplementedError
    return res




class ReplayBuffer(ReplayBufferCls):
    def push(self, experience, **kwargs):
        rate = 0.2
        index = self.size % self.capacity
        if self.size >= self.capacity:
            index = (index % int((1-rate)*self.capacity)) + rate*self.capacity
            index = int(index)

        self.memory[index] = experience
        self.size += 1



class ReplayBufferMultiWorker(object):
    buffer_cls = ReplayBuffer

    def __init__(self, config, capacity, batch_size, device):
        num_workers = config.num_workers
        self.num_workers = num_workers
        self.batch_size, self.device = batch_size, device
        buffer_cls = config.get('buffer_cls', self.buffer_cls)
        self.buffers = {i: buffer_cls(config, capacity //num_workers, batch_size, device) for i in range(num_workers)}
        return


    def __len__(self):
        lengths = [len(b) for b in self.buffers.values()]
        return sum(lengths)

    def push(self, experience, **kwargs):
        i = kwargs.get('index')
        self.buffers[i].push(experience)
        return

    def sample(self):
        batch_sizes = rllib.basic.split_integer(self.batch_size, self.num_workers)
        batch = []
        for i, buffer in self.buffers.items():
            batch.append(buffer.get_batch(batch_sizes[i]))
        batch = np.concatenate(batch)
        return self._batch_stack(batch).to(self.device)

    def _batch_stack(self, batch):
        raise NotImplementedError





class RolloutBufferMultiWorker(object):
    buffer_cls = rllib.buffer.RolloutBuffer

    def __init__(self, config: rllib.basic.YamlConfig, device, batch_size):
        num_workers = config.num_workers
        self.num_workers = num_workers
        self.batch_size, self.device = batch_size, device
        buffer_cls = config.get('buffer_cls', self.buffer_cls)
        self.buffers = {i: buffer_cls(config, device, batch_size) for i in range(num_workers)}
        return


    def __len__(self):
        lengths = [len(b) for b in self.buffers.values()]
        return sum(lengths)

    def push(self, experience, **kwargs):
        i = kwargs.get('index')
        self.buffers[i].push(experience)
        return

    def sample(self, gamma):
        [buffer.reward2return(gamma) for buffer in self.buffers.values()]
        batch_sizes = rllib.basic.split_integer(self.batch_size, self.num_workers)
        batch = []
        for i, buffer in self.buffers.items():
            batch.append(buffer.get_batch(batch_sizes[i]))
        batch = np.concatenate(batch)
        return self._batch_stack(batch).to(self.device)

    def _batch_stack(self, batch):
        raise NotImplementedError


    def clear(self):
        [buffer.clear() for buffer in self.buffers.values()]




if __name__ == '__main__':
    replay_buffer = ReplayBuffer(None, 101, 2, device='cpu')

    import pdb; pdb.set_trace()

    for i in range(206):
        replay_buffer.push(i)


    import pdb; pdb.set_trace()
