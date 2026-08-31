import numpy as np
import matplotlib.pyplot as plt


def do_nothing_action(executor):
    return [i for i, fn in executor.macro_handlers.items() if fn.__name__ == "_do_nothing"][0]


def evaluate(agent, encoder, executor, env, opponent_state_dict=None, num_episodes=5,
             max_steps=720, hidden_dim=128, deterministic=True, verbose=True):
    obs_dim, num_actions = encoder.output_dim, executor.num_macros
    opponent = None
    if opponent_state_dict is not None:
        from agent import FrozenOpponent
        opponent = FrozenOpponent(obs_dim, hidden_dim, num_actions, opponent_state_dict)

    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        h0 = agent.new_hidden()
        h1 = agent.new_hidden() if opponent is not None else None

        while not done and step < max_steps:
            step += 1
            obs0 = encoder.encode(obs[0])
            obs1 = encoder.encode(obs[1])
            mask0 = executor.legal_macros(obs[0])
            mask1 = executor.legal_macros(obs[1])

            if deterministic:
                action0, h0 = agent.act_greedy(obs0, h0, mask=mask0)
            else:
                action0, _, h0 = agent.act(obs0, h0, mask=mask0)

            if opponent is not None:
                action1, h1 = opponent.act(obs1, h1, mask=mask1)
            else:
                action1 = do_nothing_action(executor)

            next_obs, rewards, dones, _ = env.step({0: action0, 1: action1})
            done = dones[0]
            obs = next_obs

        money0 = obs[0]["farms"][0]["money"]
        money1 = obs[1]["farms"][1]["money"]
        won = money0 > money1
        results.append((money0, money1, won))
        if verbose:
            print(f"[eval ep {ep+1}/{num_episodes}] agent_money={money0:.0f} "
                  f"opponent_money={money1:.0f} agent_won={won}")

    agent_moneys = [r[0] for r in results]
    win_rate = sum(r[2] for r in results) / len(results)
    print(f"\nMean agent final money: {np.mean(agent_moneys):.1f} "
          f"(+/- {np.std(agent_moneys):.1f})  |  win rate: {win_rate:.0%}")
    return results


def plot_training_curve(history):
    iters = [h["iter"] for h in history]
    money = [h["mean_money"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(iters, money, label="mean money (self-play episodes)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("mean money")
    if "p_self_play" in history[0]:
        ax2 = ax1.twinx()
        ax2.plot(iters, [h["p_self_play"] for h in history], color="gray", alpha=0.4, label="p_self_play")
        ax2.set_ylabel("p_self_play")
    fig.tight_layout()
    plt.show()
