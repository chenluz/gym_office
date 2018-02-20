from gym.envs.registration import register

register(
    id='office_control-v0',
    entry_point='office_control.envs:OfficeEnv'
)

register(
    id='office_control-v1',
    entry_point='office_control.envs:OfficeSampleEnv'
)

register(
    id='office_control-v2',
    entry_point='office_control.envs:OfficeSimulateEnv'
)