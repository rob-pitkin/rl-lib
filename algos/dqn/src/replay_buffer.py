from collections import deque
import random
from torch import tensor


class ReplayBuffer:
    """
    Replay buffer class for a DQN agent

    Attributes:
        capacity (int): capacity of the replay buffer
        buffer (deque): replay buffer
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, experience: tuple[tensor, int, float, tensor, bool]) -> None:
        """
        Adds an element to the replay buffer

        Args:
            experience (tuple): a tuple containing the state, action, reward, next state, and if the episode has terminated/truncated

        Returns:
            None
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the replay buffer

        Args:
            batch_size (int): size of the batch to sample
        """
        return random.sample(self.buffer, batch_size)

    def getSize(self) -> int:
        """
        Get the current size of the replay buffer

        Args:
            None
        Returns:
            int: the size of the replay buffer
        """
        return len(self.buffer)
