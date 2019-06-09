# Reinforcement Learning
:+1: There is no supervisor, only a reward signal

:+1: Feedback is delayed, not instantaneous

:+1: Time really matters (sequential)

:+1: Agent's action affect the subsequent data it receives

## RL terminologies
1. Reward, is a scalar feed back signal, it indicates how weell agent is doing at step t. The agent's jobs is to maximise cumulative reward. RL is based on th ereward hypothesis
   
2. Sequential DEcision Making: 
    1. Gloal: select actions to maximise total future reward
    2. Actions may have long term consequences.
    3. Reward may be delayed
    4. It may be better to sacrifice immediate reward to gain more long-term reward.
    5. Example:
        1. Afinancial investment (may take months to mature)
        2. Refuelling a helicopter
        3. Blocking oppoent moves
        
3. Agent and Environment:
    1. At each step ``t`` the agent:
        1. excutes action ``A_t``
        2. Recieves observation ``O_t``
        3. Receives scalar reward ``R_t``
    2. The environment:
        1. Receives action ``A_t``
        2. Emits observation ``O_{t+1}``
        3. Emits scalar reward ``R_{t+1}``
    3. ``t`` increments at env.   
    
3. History and State:
    1. The history is th esequence of {obsevations, actions, rewards} ``H_t = O_1,R_1,A_1...A_{t-1},O_t,R_t``
    2. All observable variables up to time ``t``
    3. The sensorimotor stream of a robot or embodied agent.
    4. What happens next depends on the history
    5. State is the information used to determine what happens next
    6. Formally, state is a function of the history: ``S_t = f(H_t)``


4. Environment state
    1. The environment state ``S_t^e`` is the environment's private representation
    2. the environment state is not usually visible to the agent
    
5. Agent State ``S_t^a``:
    1. It's the informatin used by reinforcement learning algorithms
    2. ``S_t^a=f(H_t)`` It can be any function of history:
    
                               