import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor import CROPS, ANIMALS, MARKET_BASE_PRICE, FERTILIZER_BASE_PRICE


def crop_in_ground_value(tile, prices, day):
    """Approximate fair value of a not-yet-harvested crop: linearly
    interpolate between 0 (just planted) and the full market value of its
    expected max yield, based on progress toward first_yield_day."""
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return 0.0
    crop = tile.get("crop")
    info = CROPS.get(crop)
    if info is None:
        return 0.0
    planted_day = tile.get("planted_day", day)
    progress = (day - planted_day) / max(info["first_yield_day"], 1)
    progress = max(0.0, min(1.0, progress))
    expected_value = info["max_yield"] * prices.get(crop, MARKET_BASE_PRICE.get(crop, 0))
    return progress * expected_value


def animal_value(tile, day):
    """Approximate fair value of an owned animal: purchase cost, discounted
    if currently at risk of escaping (unfed today)."""
    if not (isinstance(tile, dict) and tile.get("animal")):
        return 0.0
    animal = tile["animal"]
    info = ANIMALS.get(animal)
    if info is None:
        return 0.0
    at_risk = not tile.get("fed_today", False)
    return info["cost"] * (0.6 if at_risk else 1.0)


def farm_asset_potential(farm, private, prices, day):
    """Extended potential: money + shed + growing crops + animals."""
    money = farm["money"]
    shed = private["shed"]
    shed_value = sum(prices.get(item, 0) * shed.get(item, 0) for item in farm.get("sellable_items", []))

    crop_value = 0.0
    animal_val = 0.0
    for row in farm["tiles"]:
        for tile in row:
            crop_value += crop_in_ground_value(tile, prices, day)
            animal_val += animal_value(tile, day)

    return money + shed_value + crop_value + animal_val


def _get_obs(obs, player):
    """Handle both struct and dict obs formats."""
    if isinstance(obs, list):
        return obs[player]
    return obs


def relative_potential(obs, player, day):
    """Compute relative potential for a player vs opponent.
    Formula: (my_money - opp_money) + (my_assets - opp_assets)
    """
    my_farm = _get_obs(obs, player)["farms"][player]
    opp_farm = _get_obs(obs, 1 - player)["farms"][1 - player]
    
    my_money = my_farm["money"]
    opp_money = opp_farm["money"]
    
    my_shed = _get_obs(obs, player)["private"]["shed"]
    opp_shed = _get_obs(obs, 1 - player)["private"]["shed"]
    
    prices = _get_obs(obs, player)["market"]["prices"]
    
    my_shed_value = sum(prices.get(item, 0) * my_shed.get(item, 0) for item in my_farm.get("sellable_items", []))
    opp_shed_value = sum(prices.get(item, 0) * opp_shed.get(item, 0) for item in opp_farm.get("sellable_items", []))
    
    my_crop_value = 0.0
    opp_crop_value = 0.0
    my_animal_val = 0.0
    opp_animal_val = 0.0
    
    for row in my_farm["tiles"]:
        for tile in row:
            my_crop_value += crop_in_ground_value(tile, prices, day)
            my_animal_val += animal_value(tile, day)
    
    for row in opp_farm["tiles"]:
        for tile in row:
            opp_crop_value += crop_in_ground_value(tile, prices, day)
            opp_animal_val += animal_value(tile, day)
    
    my_potential = my_money + my_shed_value + my_crop_value + my_animal_val
    opp_potential = opp_money + opp_shed_value + opp_crop_value + opp_animal_val
    
    return my_potential - opp_potential
