# office_control

There are four different types of environment in the fold:

Type 1: sample environment, in which environment is just sample data
Type 2: simulated environment, in which environment is simulated by certain functions.
Type 3: a real chamber environment, in which action is turn on/off fan, fan heater, and air conditioner
Type 4: a real office-like environment

Type 4 is the one that is in used.  This environment is explained below in detail:

6 occupants
6 desk
Observation from envionrmental sensor and Microsoft band
Action is turn on/off six fan heater through Plugwise

#### Example

```python
import gym;
import office_control.envs as office_env;

env = gym.make('office_control-v0');

