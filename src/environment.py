from encoder import ObservationEncoder
from executor import MacroActionExecutor
from values import farm_asset_potential


class KaggricultureEnv:
    SELLABLE_ITEMS = ObservationEncoder.SELLABLE_ITEMS

    def __init__(self, executor=None, debug=False, shaping_gamma=1.0,
                 reward_config=None):
        from kaggle_environments import make
        self.env = make("kaggriculture", debug=debug)
        self.num_players = 2
        self.executor = executor
        self.shaping_gamma = shaping_gamma
        
        default_reward_config = {
            "invalid_penalty": -20.0,
            "do_nothing_penalty": -10.0,
            "do_nothing_threshold": 2,
            "terminal_win_bonus": 2000.0,
            "terminal_loss_penalty": -1000.0,
            "terminal_tie_bonus": 500.0,
            "effectiveness_bonus": {
                "harvest": 1.0,
                "sell": 0.5,
                "plant": 0.3,
                "water": 0.2,
                "feed": 0.3,
            },
            "shaped_weight": 1.0,
        }
        
        self.reward_config = default_reward_config.copy()
        if reward_config:
            for key in reward_config:
                if key == "effectiveness_bonus":
                    self.reward_config["effectiveness_bonus"].update(reward_config["effectiveness_bonus"])
                else:
                    self.reward_config[key] = reward_config[key]
        
        self.reset()

    def reset(self):
        self.env.reset()
        self._last_obs = [self.env.steps[0][i].observation for i in range(self.num_players)]
        self.initial_money = {p: self._last_obs[p]["farms"][p]["money"] for p in range(self.num_players)}
        self._prev_potential = [self._compute_potential(obs) for obs in self._last_obs]
        self._prev_legal_actions = [None, None]
        return self._last_obs

    def step(self, actions):
        primitive_actions = {}
        for p in range(self.num_players):
            action = actions[p]
            if isinstance(action, int) and self.executor is not None:
                primitive_actions[p] = self.executor.execute(self._last_obs[p], int(action))
            elif isinstance(action, dict):
                primitive_actions[p] = action
            else:
                raise ValueError(f"Invalid action type for player {p}")
        self.env.step([primitive_actions[0], primitive_actions[1]])
        next_obs = [self.env.steps[-1][i].observation for i in range(self.num_players)]
        rewards = {}
        for p in range(self.num_players):
            reward = self._compute_reward(
                self._last_obs[p], next_obs[p], actions[p],
                self._prev_legal_actions[p], self.env.done
            )
            rewards[p] = reward
            self._prev_potential[p] = self._compute_potential(next_obs[p], is_terminal=self.env.done)
            self._prev_legal_actions[p] = self.executor.legal_macros(next_obs[p]) if self.executor else None
        dones = {p: self.env.done for p in range(self.num_players)}
        self._last_obs = next_obs
        return next_obs, rewards, dones, None

    def _compute_reward(self, prev_obs, next_obs, action, prev_legal, is_terminal=False):
        my_money = next_obs["farms"][0]["money"]
        opp_money = next_obs["farms"][1]["money"]
        my_prev_money = prev_obs["farms"][0]["money"]
        opp_prev_money = prev_obs["farms"][1]["money"]
        
        delta_relative = (my_money - opp_money) - (my_prev_money - opp_prev_money)
        
        if is_terminal:
            if my_money > opp_money:
                delta_relative += self.reward_config["terminal_win_bonus"]
            elif my_money < opp_money:
                delta_relative += self.reward_config["terminal_loss_penalty"]
            else:
                delta_relative += self.reward_config["terminal_tie_bonus"]
            return delta_relative
        
        if prev_legal is not None and isinstance(action, int):
            if not prev_legal[action]:
                delta_relative += self.reward_config["invalid_penalty"]
        
        if isinstance(action, int) and action == 0:
            profit_actions = [a for a in range(18) if prev_legal is not None and prev_legal[a] and a in {2, 3, 9, 10, 11, 12, 13, 5}]
            if len(profit_actions) >= self.reward_config["do_nothing_threshold"]:
                delta_relative += self.reward_config["do_nothing_penalty"]
        
        if isinstance(action, int) and self.executor is not None:
            prev_money = my_prev_money
            curr_money = my_money
            if curr_money > prev_money:
                delta_relative += self.reward_config["effectiveness_bonus"]["sell"]
            
            prev_shed = prev_obs["private"]["shed"]
            curr_shed = next_obs["private"]["shed"]
            for item in curr_shed:
                if curr_shed[item] > prev_shed.get(item, 0):
                    delta_relative += self.reward_config["effectiveness_bonus"]["harvest"]
        
        shaped_potential = self._compute_potential(next_obs) - self._compute_potential(prev_obs)
        shaped_weight = self.reward_config.get("shaped_weight", 1.0)
        
        return shaped_weight * shaped_potential + (1 - shaped_weight) * delta_relative

    def _compute_potential(self, obs, is_terminal=False):
        player = obs["player"]
        money = obs["farms"][player]["money"]
        if is_terminal:
            return money
        shed = obs["private"]["shed"]
        prices = obs["market"]["prices"]
        shed_value = sum(prices.get(item, 0) * shed.get(item, 0) for item in self.SELLABLE_ITEMS)
        return money + shed_value
