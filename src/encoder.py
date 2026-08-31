import numpy as np

class ObservationEncoder:
    CROP_TYPES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
    ANIMAL_TYPES = ["GOOSE", "COW", "SHEEP"]
    SELLABLE_ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                      "EGG", "MILK", "WOOL", "FERTILIZER"]
    SHOP_TYPES = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
                  "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]

    def __init__(self, board_size=10, include_opponent=True):
        self.board_size = board_size
        self.include_opponent = include_opponent
        self.tile_vec_len = 25
        self.market_town_vec_len = 9 + 9 + 30 + len(self.SHOP_TYPES)
        self.private_vec_len = 9 + 5 + 4 + 1 + 1

    def encode_tile(self, tile):
        vec = np.zeros(self.tile_vec_len, dtype=np.float32)
        if tile is None:
            vec[0] = 1.0
        elif tile == "LOCKED":
            vec[1] = 1.0
        elif isinstance(tile, dict):
            kind = tile.get("kind")
            if kind == "PLANT":
                vec[2] = 1.0
                crop = tile.get("crop", "")
                if crop in self.CROP_TYPES:
                    vec[3 + self.CROP_TYPES.index(crop)] = 1.0
                vec[8] = tile.get("planted_day", 0) / 30.0
                vec[9] = float(tile.get("watered_today", False))
                vec[10] = tile.get("yield_units", 0) / 6.0
                vec[11] = tile.get("fertilized_until_day", -1) / 30.0
                vec[12] = tile.get("consecutive_unwatered", 0) / 2.0
            elif kind == "WEED":
                vec[13] = 1.0
            elif kind in ("COOP", "PASTURE"):
                vec[14] = 1.0 if kind == "COOP" else 0.0
                vec[15] = 1.0 if kind == "PASTURE" else 0.0
                animal = tile.get("animal")
                if animal is None:
                    vec[16] = 1.0
                else:
                    if animal in self.ANIMAL_TYPES:
                        vec[17 + self.ANIMAL_TYPES.index(animal)] = 1.0
                vec[20] = float(tile.get("fed_today", False))
                vec[21] = float(tile.get("cared_today", False))
                vec[22] = float(tile.get("fertilizer_available", False))
                vec[23] = tile.get("yield_units", 0) / 6.0
                vec[24] = tile.get("days_to_next_yield", 10) / 10.0
        return vec

    def encode_market_town(self, obs):
        market = obs["market"]
        town = obs["town"]

        prices = np.zeros(9, dtype=np.float32)
        base_prices = np.zeros(9, dtype=np.float32)
        for i, item in enumerate(self.SELLABLE_ITEMS):
            prices[i] = market["prices"].get(item, 0) / 100.0
            base_prices[i] = prices[i]

        day_vec = np.zeros(30, dtype=np.float32)
        day = obs["day"]
        day_vec[day - 1] = 1.0

        shop_vec = np.zeros(len(self.SHOP_TYPES), dtype=np.float32)
        for shop_type in town.get("unlocked_shops", []):
            if shop_type in self.SHOP_TYPES:
                idx = self.SHOP_TYPES.index(shop_type)
                shop_vec[idx] = 1.0

        return np.concatenate([prices, base_prices, day_vec, shop_vec])

    def encode_private(self, private):
        money = private.get("money", 3000) / 1000.0
        seeds = np.zeros(5, dtype=np.float32)
        for i, crop in enumerate(self.CROP_TYPES):
            seeds[i] = private["seeds"].get(crop, 0) / 10.0
        shed = np.zeros(9, dtype=np.float32)
        for i, item in enumerate(self.SELLABLE_ITEMS):
            shed[i] = private["shed"].get(item, 0) / 10.0
        inventory = np.zeros(4, dtype=np.float32)
        inv_list = private.get("inventories", [{}])
        if inv_list:
            first_inv = inv_list[0]
            inventory[0] = first_inv.get("WHEAT", 0) / 10.0
            inventory[1] = first_inv.get("CARROT", 0) / 10.0
            inventory[2] = first_inv.get("TOMATO", 0) / 10.0
            inventory[3] = first_inv.get("FERTILIZER", 0) / 10.0
        hires = min(private.get("hires_today", 0), 5) / 5.0

        return np.concatenate([[money], seeds, shed, inventory, [hires]])

    def encode(self, obs):
        farm = obs["farms"][obs["player"]]
        tiles = []
        for row in farm["tiles"]:
            for tile in row:
                tiles.append(self.encode_tile(tile))
        tiles = np.stack(tiles)

        market_town = self.encode_market_town(obs)
        private = self.encode_private(obs["private"])

        if self.include_opponent:
            opp_farm = obs["farms"][1 - obs["player"]]
            opp_tiles = []
            for row in opp_farm["tiles"]:
                for tile in row:
                    opp_tiles.append(self.encode_tile(tile))
            opp_tiles = np.stack(opp_tiles)
            tiles = np.concatenate([tiles, opp_tiles], axis=0)

        return np.concatenate([tiles.flatten(), market_town, private])

    @property
    def output_dim(self):
        tile_count = self.board_size * self.board_size
        if self.include_opponent:
            tile_count *= 2
        return tile_count * self.tile_vec_len + self.market_town_vec_len + self.private_vec_len

    def encode_global(self, obs, perspective=0):
        farm = obs[0]["farms"][perspective]
        private = obs[0]["private"]
        money = farm["money"] / 1000.0
        shed = private["shed"]
        prices = obs[0]["market"]["prices"]
        shed_value = sum(prices.get(item, 0) * qty for item, qty in shed.items())
        return np.array([money, shed_value], dtype=np.float32)
