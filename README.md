# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## POLICY ITERATION ALGORITHM
The policy iteration algorithm is a method for finding the optimal policy in a Markov Decision Process (MDP). It alternates between policy evaluation (finding the value function for a fixed policy) and policy improvement (updating the policy using the new value function).

## VALUE ITERATION ALGORITHM
- Value iteration is a method of computing an optimal MDP policy  and its value.
- It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
- The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:
1. Initialize the value function `V(s)` arbitrarily for all states `s`.
2. Repeat until convergence:
    - Initialize aaction-value function `Q(s, a)` arbitrarily for all states `s` and actions `a`.
    - For all the states s and all the action a of every state:
        - Update the action-value function `Q(s, a)` using the Bellman equation.
        - Take the value function `V(s)` to be the maximum of `Q(s, a)` over all actions `a`.
        - Check if the maximum difference between `Old V` and `new V` is less than `theta`, where theta is a **small positive number** that determines the **accuracy of estimation**.
3. If the maximum difference between Old V and new V is greater than theta, then
    - Update the value function `V` with the **maximum action-value** from `Q`.
    - Go to **step 2**.
4. The optimal policy can be constructed by taking the **argmax** of the action-value function `Q(s, a)` over all actions `a`.
5. Return the optimal policy and the optimal value function.
## VALUE ITERATION FUNCTION
### Name:MITHUN M S
### Register Number: 212222240067
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, s_prime, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[s_prime] * (1.0 - done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
        pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/5780db5f-229d-4eaa-bd7c-3267324fae13)

![image](https://github.com/user-attachments/assets/f0c0025a-699e-4ac3-933e-b301b303d91b)

![image](https://github.com/user-attachments/assets/b30207a6-ab10-4ed1-b874-f6a54b1c465c)



## RESULT:

thus we successfully developed Python program to find the optimal policy for the given MDP using the value iteration algorithm.
