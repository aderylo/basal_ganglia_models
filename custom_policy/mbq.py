




from tianshou.policy import BasePolicy, DQNPolicy, PGPolicy


class MBQPolicy(BasePolicy):
    def __init__(
        value_based_policy: DQNPolicy,
        temperature_net
    )
