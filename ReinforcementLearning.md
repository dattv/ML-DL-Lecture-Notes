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
        
        