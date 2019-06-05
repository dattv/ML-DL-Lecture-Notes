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
                  