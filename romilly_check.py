"""How long were Cooper, Brand and Doyle on Miller's planet, given the computed time dilation?"""
import json
D = json.load(open("results/gargantua_system.json"))["dilation"]["Miller's planet"]
earth_years = 23 + 4 / 12 + 8 / 365.25                      # what passed for Romilly
planet_hours = earth_years * 365.25 * 24 / D
out = {"dilation": D, "romilly_years": earth_years, "hours_on_planet": planet_hours,
       "minutes_on_planet": planet_hours * 60,
       "earth_time_per_planet_minute_days": D / 60 / 24}
print(out); json.dump(out, open("results/romilly_check.json", "w"), indent=1)
