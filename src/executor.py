"""
MacroActionExecutor v2.
Verified directly against the installed kaggle_environments source
(envs/kaggriculture/kaggriculture.py) rather than assumed.
"""

import numpy as np

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
}

ANIMALS = {
    "GOOSE": {"cost": 150, "structure": "COOP", "product": "EGG", "yield_per": 1, "feed_per": 1},
    "COW":   {"cost": 300, "structure": "PASTURE", "product": "MILK", "yield_per": 1, "feed_per": 2},
    "SHEEP": {"cost": 250, "structure": "PASTURE", "product": "WOOL", "yield_per": 1, "feed_per": 2},
}

MARKET_BASE_PRICE = {
    "WHEAT": 10, "CARROT": 15, "TOMATO": 20, "STRAWBERRY": 25, "MELON": 30,
    "EGG": 20, "MILK": 25, "WOOL": 30,
}

FERTILIZER_BASE_PRICE = 50


class MacroActionExecutor:
    def __init__(self, board_size=10, sell_price_floor_frac=0.8):
        self.board_size = board_size
        self.sell_price_floor_frac = sell_price_floor_frac
        self.num_macros = 18

        self.macro_handlers = {
            0: self._do_nothing,
            1: self._water_all,
            2: self._harvest_ready,
            3: self._sell_ready,
            4: self._hire_hand,
            5: self._buy_land,
            6: self._feed_animals,
            7: self._care_animals,
            8: self._fertilize_ready,
            9: self._make_plant_macro("WHEAT"),
            10: self._make_plant_macro("CARROT"),
            11: self._make_plant_macro("TOMATO"),
            12: self._make_plant_macro("STRAWBERRY"),
            13: self._make_plant_macro("MELON"),
            14: self._make_buy_place_animal_macro("GOOSE"),
            15: self._make_buy_place_animal_macro("COW"),
            16: self._make_buy_place_animal_macro("SHEEP"),
            17: self._buy_fertilizer,
        }

    def execute(self, obs, macro_id):
        handler = self.macro_handlers.get(macro_id)
        if handler is None:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return handler(obs)

    def legal_macros(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        money = farm["money"]
        day = obs["day"]

        def any_tile(cond):
            for row in farm["tiles"]:
                for tile in row:
                    if cond(tile):
                        return True
            return False

        mask = np.zeros(self.num_macros, dtype=bool)
        for i, fn in self.macro_handlers.items():
            name = fn.__name__
            if name == "_do_nothing":
                mask[i] = True
            elif name == "_water_all":
                mask[i] = any_tile(self._is_unwatered)
            elif name == "_harvest_ready":
                mask[i] = any_tile(self._is_harvestable)
            elif name == "_sell_ready":
                shed = private["shed"]
                prices = obs["market"]["prices"]
                mask[i] = any(
                    shed.get(item, 0) > 0 and
                    prices.get(item, 0) >= self.sell_price_floor_frac * MARKET_BASE_PRICE.get(item, 0)
                    for item in MARKET_BASE_PRICE
                )
            elif name == "_hire_hand":
                n = farm["hires_today"]
                mask[i] = n < 5 and money >= self._hire_cost(n)
            elif name == "_buy_land":
                land_order = ["NE", "SW", "SE"]
                land_prices = [1000, 2000, 4000]
                n_extra = len(farm["unlocked_quadrants"]) - 1
                mask[i] = n_extra < len(land_order) and money >= land_prices[n_extra]
            elif name == "_feed_animals":
                mask[i] = any_tile(lambda t: self._is_animal(t) and not t.get("fed_today", False))
            elif name == "_care_animals":
                mask[i] = any_tile(lambda t: self._is_animal(t) and not t.get("cared_today", False))
            elif name == "_fertilize_ready":
                has_fert = private.get("inventories", [{}])[0].get("FERTILIZER", 0) > 0
                needs_it = any_tile(lambda t: self._is_plant(t) and t.get("fertilized_until_day", -1) < day)
                has_ready = any_tile(lambda t: self._is_animal(t) and t.get("fertilizer_available", False))
                mask[i] = (has_fert and needs_it) or has_ready
            elif name.startswith("_plant_"):
                crop = name.replace("_plant_", "").upper()
                seeds = private["seeds"].get(crop, 0)
                can_afford_one = money >= CROPS[crop]["seed"]
                has_empty_tile = any_tile(self._is_empty)
                mask[i] = has_empty_tile and (seeds > 0 or can_afford_one)
            elif name.startswith("_buy_place_"):
                animal = name.replace("_buy_place_", "").upper()
                carrying = private.get("inventories", [{}])[0].get(animal, 0) > 0
                in_shed = private["shed"].get(animal, 0) > 0
                can_afford = money >= ANIMALS[animal]["cost"]
                mask[i] = carrying or in_shed or can_afford
            elif name == "_buy_fertilizer":
                mask[i] = money >= FERTILIZER_BASE_PRICE
            else:
                mask[i] = True

        if not mask.any():
            mask[0] = True
        return mask

    def _pos_to_dir(self, fx, fy, tx, ty):
        if fx < tx: return "EAST"
        if fx > tx: return "WEST"
        if fy < ty: return "SOUTH"
        if fy > ty: return "NORTH"
        return None

    def _nearest_tile(self, farm, condition, from_pos, exclude=None):
        exclude = exclude or set()
        best, best_dist = None, 10**9
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x, y) in exclude:
                    continue
                tile = farm["tiles"][y][x]
                if condition(tile):
                    dist = abs(x - from_pos[0]) + abs(y - from_pos[1])
                    if dist < best_dist:
                        best, best_dist = (x, y), dist
        return best

    def _is_plant(self, tile):
        return isinstance(tile, dict) and tile.get("kind") == "PLANT"

    def _is_animal(self, tile):
        return isinstance(tile, dict) and "animal" in tile

    def _is_unwatered(self, tile):
        return self._is_plant(tile) and not tile.get("watered_today", False)

    def _is_harvestable(self, tile):
        if self._is_plant(tile) and tile.get("yield_units", 0) > 0:
            return True
        if self._is_animal(tile) and tile.get("yield_units", 0) > 0:
            return True
        return False

    def _is_empty(self, tile):
        return tile is None

    def _units(self, farm):
        hands = farm.get("hands", [])
        return [("farmer", tuple(farm["farmer"]))] + [("hand", tuple(h)) for h in hands]

    def _route_units(self, farm, units, is_target, on_target_action):
        claimed = set()
        farmer_action = ["PASS"]
        hand_actions = []
        for unit_type, pos in units:
            fx, fy = pos
            tile = farm["tiles"][fy][fx]
            if (fx, fy) not in claimed and is_target(tile):
                action = list(on_target_action)
                claimed.add((fx, fy))
            else:
                target = self._nearest_tile(farm, is_target, (fx, fy), exclude=claimed)
                if target:
                    claimed.add(target)
                    step = self._pos_to_dir(fx, fy, target[0], target[1])
                    action = [step] if step else ["PASS"]
                else:
                    action = ["PASS"]
            if unit_type == "farmer":
                farmer_action = action
            else:
                hand_actions.append(action)
        return farmer_action, hand_actions

    def _shed_adjacent_tiles(self):
        half = self.board_size // 2
        return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}

    def _do_nothing(self, obs):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def _make_plant_macro(self, crop):
        def macro(obs):
            player = obs["player"]
            farm = obs["farms"][player]
            private = obs["private"]
            seeds = private["seeds"].get(crop, 0)
            money = farm["money"]
            market = []
            units = self._units(farm)

            if seeds == 0:
                seed_cost = CROPS[crop]["seed"]
                if money >= seed_cost:
                    market.append(["BUY_SEED", crop, 1])
                return {"farmer": ["PASS"], "hands": [], "market": market}

            plant_budget = [seeds]

            claimed = set()
            farmer_action = ["PASS"]
            hand_actions = []

            for unit_type, pos in units:
                fx, fy = pos
                tile = farm["tiles"][fy][fx]

                if (fx, fy) not in claimed and tile is None and plant_budget[0] > 0:
                    action = ["PLANT", crop]
                    plant_budget[0] -= 1
                    claimed.add((fx, fy))
                elif tile is None and plant_budget[0] <= 0:
                    action = ["PASS"]
                else:
                    target = self._nearest_tile(farm, self._is_empty, (fx, fy), exclude=claimed)
                    if target:
                        claimed.add(target)
                        step = self._pos_to_dir(fx, fy, target[0], target[1])
                        action = [step] if step else ["PASS"]
                    else:
                        action = ["PASS"]

                if unit_type == "farmer":
                    farmer_action = action
                else:
                    hand_actions.append(action)

            return {"farmer": farmer_action, "hands": hand_actions, "market": market}

        macro.__name__ = f"_plant_{crop.lower()}"
        return macro

    def _water_all(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        units = self._units(farm)
        farmer_action, hand_actions = self._route_units(farm, units, self._is_unwatered, ["WATER"])
        return {"farmer": farmer_action, "hands": hand_actions, "market": []}

    def _harvest_ready(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        units = self._units(farm)
        farmer_action, hand_actions = self._route_units(farm, units, self._is_harvestable, ["HARVEST"])
        return {"farmer": farmer_action, "hands": hand_actions, "market": []}

    def _sell_ready(self, obs):
        shed = obs["private"]["shed"]
        prices = obs["market"]["prices"]
        market = []
        for item, qty in shed.items():
            if qty <= 0:
                continue
            base = MARKET_BASE_PRICE.get(item)
            price = prices.get(item, 0)
            if base and price < self.sell_price_floor_frac * base:
                continue
            market.append(["SELL", item, int(qty)])
        return {"farmer": ["PASS"], "hands": [], "market": market}

    def _hire_hand(self, obs):
        farm = obs["farms"][obs["player"]]
        n_hired = farm["hires_today"]
        if n_hired >= 5:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        cost = self._hire_cost(n_hired)
        if farm["money"] >= cost:
            return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    @staticmethod
    def _hire_cost(n_already_today, mult=1):
        a, b = 1, 1
        for _ in range(n_already_today):
            a, b = b, a + b
        return mult * a

    def _buy_land(self, obs):
        farm = obs["farms"][obs["player"]]
        land_order = ["NE", "SW", "SE"]
        land_prices = [1000, 2000, 4000]
        n_extra = len(farm["unlocked_quadrants"]) - 1
        if n_extra >= len(land_order):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        if farm["money"] >= land_prices[n_extra]:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def _feed_animals(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        units = self._units(farm)

        def needs_feed(tile):
            return self._is_animal(tile) and not tile.get("fed_today", False)

        claimed = set()
        farmer_action = ["PASS"]
        hand_actions = []
        for i, (unit_type, pos) in enumerate(units):
            fx, fy = pos
            tile = farm["tiles"][fy][fx]
            inv = private["inventories"][i] if i < len(private.get("inventories", [])) else {}
            if (fx, fy) not in claimed and needs_feed(tile):
                if inv.get("WHEAT", 0) > 0:
                    action = ["FEED"]
                elif (fx, fy) in self._shed_adjacent_tiles() and private["shed"].get("WHEAT", 0) > 0:
                    action = ["PICKUP", "WHEAT", 1]
                else:
                    action = ["PASS"]
                claimed.add((fx, fy))
            else:
                target = self._nearest_tile(farm, needs_feed, (fx, fy), exclude=claimed)
                if target:
                    claimed.add(target)
                    step = self._pos_to_dir(fx, fy, target[0], target[1])
                    action = [step] if step else ["PASS"]
                else:
                    action = ["PASS"]
            if unit_type == "farmer":
                farmer_action = action
            else:
                hand_actions.append(action)
        return {"farmer": farmer_action, "hands": hand_actions, "market": []}

    def _care_animals(self, obs):
        farm = obs["farms"][obs["player"]]
        units = self._units(farm)

        def needs_care(tile):
            return self._is_animal(tile) and not tile.get("cared_today", False)

        farmer_action, hand_actions = self._route_units(farm, units, needs_care, ["CARE"])
        return {"farmer": farmer_action, "hands": hand_actions, "market": []}

    def _fertilize_ready(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        day = obs.get("day", 0)
        units = self._units(farm)

        def needs_fertilizer(tile):
            return self._is_plant(tile) and tile.get("fertilized_until_day", -1) < day

        def has_fertilizer_ready(tile):
            return self._is_animal(tile) and tile.get("fertilizer_available", False)

        claimed = set()
        farmer_action = ["PASS"]
        hand_actions = []
        for i, (unit_type, pos) in enumerate(units):
            fx, fy = pos
            tile = farm["tiles"][fy][fx]
            inv = private["inventories"][i] if i < len(private.get("inventories", [])) else {}
            if (fx, fy) not in claimed and needs_fertilizer(tile) and inv.get("FERTILIZER", 0) > 0:
                action = ["FERTILIZE"]
                claimed.add((fx, fy))
            elif (fx, fy) not in claimed and has_fertilizer_ready(tile):
                action = ["COLLECT_FERTILIZER"]
                claimed.add((fx, fy))
            else:
                if inv.get("FERTILIZER", 0) > 0:
                    target = self._nearest_tile(farm, needs_fertilizer, (fx, fy), exclude=claimed)
                else:
                    target = self._nearest_tile(farm, has_fertilizer_ready, (fx, fy), exclude=claimed)
                if target:
                    claimed.add(target)
                    step = self._pos_to_dir(fx, fy, target[0], target[1])
                    action = [step] if step else ["PASS"]
                else:
                    action = ["PASS"]
            if unit_type == "farmer":
                farmer_action = action
            else:
                hand_actions.append(action)
        return {"farmer": farmer_action, "hands": hand_actions, "market": []}

    def _buy_fertilizer(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        money = farm["money"]
        if money >= FERTILIZER_BASE_PRICE:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_FERTILIZER", 1]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def _make_buy_place_animal_macro(self, animal):
        structure = ANIMALS[animal]["structure"]
        cost = ANIMALS[animal]["cost"]

        def macro(obs):
            player = obs["player"]
            farm = obs["farms"][player]
            private = obs["private"]
            money = farm["money"]
            shed_stock = private["shed"].get(animal, 0)
            market = []

            fx, fy = farm["farmer"]
            farmer_inv = private["inventories"][0] if private.get("inventories") else {}

            if farmer_inv.get(animal, 0) > 0:
                def is_free_structure(tile):
                    return isinstance(tile, dict) and tile.get("kind") == structure and "animal" not in tile

                current_tile = farm["tiles"][fy][fx]
                if is_free_structure(current_tile):
                    return {"farmer": ["PLACE", animal], "hands": [], "market": []}
                target = self._nearest_tile(farm, is_free_structure, (fx, fy))
                if target:
                    step = self._pos_to_dir(fx, fy, target[0], target[1])
                    if step:
                        return {"farmer": [step], "hands": [], "market": []}
                if farm["tiles"][fy][fx] is None:
                    build_op = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
                    return {"farmer": [build_op], "hands": [], "market": []}
                empty = self._nearest_tile(farm, self._is_empty, (fx, fy))
                if empty:
                    step = self._pos_to_dir(fx, fy, empty[0], empty[1])
                    if step:
                        return {"farmer": [step], "hands": [], "market": []}
                return {"farmer": ["PASS"], "hands": [], "market": []}

            if shed_stock > 0:
                if (fx, fy) in self._shed_adjacent_tiles():
                    return {"farmer": ["PICKUP", animal, 1], "hands": [], "market": []}
                shed_tiles = self._shed_adjacent_tiles()
                best, best_dist = None, 10**9
                for (tx, ty) in shed_tiles:
                    d = abs(tx - fx) + abs(ty - fy)
                    if d < best_dist:
                        best, best_dist = (tx, ty), d
                if best:
                    step = self._pos_to_dir(fx, fy, best[0], best[1])
                    if step:
                        return {"farmer": [step], "hands": [], "market": []}
                return {"farmer": ["PASS"], "hands": [], "market": []}

            if money >= cost:
                market.append(["BUY_ANIMAL", animal, 1])
            return {"farmer": ["PASS"], "hands": [], "market": market}

        macro.__name__ = f"_buy_place_{animal.lower()}"
        return macro
