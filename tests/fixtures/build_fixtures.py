"""Builds tests/fixtures/RAW_recipes.csv and RAW_interactions.csv from the hand-written rows below.

The CSVs are committed; rerun only after editing this file:  python tests/fixtures/build_fixtures.py

Edge cases -- ids are referenced by tests/test_load.py:
    minutes = 0 ............... 103 (no-cook beverage), 105 (cooked stew: suspicious zero)
    minutes very large ........ 106 (2147483647 = int32 max), 107 (1051200)
    n_steps = 0 ............... 110 (steps == [])
    missing name .............. 108
    missing description ....... 109
    recipe with no reviews .... 111
    rating = 0 ................ 4 rows (users 1003, 1006, 1009, 1012)
    duplicate interaction ..... user 1002 x recipe 101 (exact duplicate row)
    single-review user ........ 1015
    missing review text ....... user 1011 x recipe 116
    high-calorie recipe ....... 119 (calories 4520.3)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# id, name, minutes, contributor_id, submitted, tags, nutrition(7), steps, description, ingredients
RECIPES = [
    (
        101,
        "classic spaghetti marinara",
        40,
        501,
        "2003-02-11",
        ["60-minutes-or-less", "main-dish", "pasta", "italian"],
        [410.2, 18.0, 30.0, 22.0, 25.0, 9.0, 21.0],
        ["boil the spaghetti", "simmer tomatoes with garlic", "toss pasta with sauce"],
        "a simple weeknight marinara",
        ["spaghetti", "tomatoes", "garlic", "olive oil", "basil", "salt"],
    ),
    (
        102,
        "garlic butter pasta",
        20,
        502,
        "2004-06-01",
        ["30-minutes-or-less", "main-dish", "pasta"],
        [520.0, 40.0, 4.0, 15.0, 30.0, 70.0, 22.0],
        ["cook pasta", "melt butter with garlic", "combine and add cheese"],
        "rich and quick",
        ["spaghetti", "butter", "garlic", "parmesan cheese", "black pepper"],
    ),
    (
        103,
        "iced lemon water",
        0,
        503,
        "2005-07-19",
        ["beverages", "no-cook", "easy"],
        [12.0, 0.0, 8.0, 0.0, 0.0, 0.0, 1.0],
        ["stir lemon juice into cold water with ice"],
        "refreshing summer drink",
        ["water", "lemon juice", "ice"],
    ),
    (
        104,
        "fresh fruit salad",
        10,
        504,
        "2006-05-02",
        ["15-minutes-or-less", "no-cook", "desserts", "fruit"],
        [150.5, 1.0, 90.0, 0.0, 3.0, 0.0, 12.0],
        ["chop the fruit", "toss with honey"],
        "colourful and healthy",
        ["strawberries", "banana", "apple", "honey"],
    ),
    (
        105,
        "hearty beef stew",
        0,
        505,
        "2007-11-23",
        ["main-dish", "beef", "stews"],
        [610.0, 45.0, 20.0, 40.0, 95.0, 60.0, 9.0],
        ["brown the beef", "add vegetables and stock", "simmer for two hours"],
        "a winter classic",
        ["beef", "potatoes", "carrots", "onion", "beef broth", "salt"],
    ),
    (
        106,
        "slow aged plum wine",
        2147483647,
        506,
        "2008-03-14",
        ["beverages", "alcoholic"],
        [120.0, 0.0, 60.0, 0.0, 0.0, 0.0, 6.0],
        ["combine plums and sugar", "ferment", "age until ready"],
        "patience required",
        ["plums", "sugar", "yeast", "water"],
    ),
    (
        107,
        "overnight sourdough bread",
        1051200,
        507,
        "2009-09-09",
        ["breads", "baking"],
        [210.0, 2.0, 1.0, 18.0, 14.0, 1.0, 14.0],
        ["feed the starter", "mix the dough", "proof", "bake"],
        "keeps the starter alive",
        ["flour", "water", "salt", "sourdough starter"],
    ),
    (
        108,
        None,
        35,
        508,
        "2010-01-30",
        ["main-dish", "chicken"],
        [450.0, 25.0, 6.0, 30.0, 80.0, 20.0, 5.0],
        ["season the chicken", "roast"],
        "the name was lost",
        ["chicken", "garlic", "lemon juice", "olive oil", "salt"],
    ),
    (
        109,
        "tomato basil soup",
        30,
        509,
        "2010-10-10",
        ["30-minutes-or-less", "soups-stews", "vegetarian"],
        [180.0, 10.0, 25.0, 30.0, 8.0, 12.0, 6.0],
        ["saute onion and garlic", "add tomatoes", "blend with basil"],
        None,
        ["tomatoes", "onion", "garlic", "basil", "vegetable broth"],
    ),
    (
        110,
        "instant banana snack",
        5,
        510,
        "2011-04-04",
        ["15-minutes-or-less", "snacks"],
        [105.0, 0.0, 30.0, 0.0, 2.0, 0.0, 9.0],
        [],
        "steps were never written",
        ["banana"],
    ),
    (
        111,
        "lonely cabbage rolls",
        90,
        511,
        "2012-12-12",
        ["main-dish", "eastern-european"],
        [330.0, 15.0, 12.0, 25.0, 40.0, 18.0, 10.0],
        ["blanch cabbage", "fill with rice and meat", "bake in sauce"],
        "nobody has reviewed this yet",
        ["cabbage", "rice", "ground beef", "tomatoes", "onion"],
    ),
    (
        112,
        "chicken curry",
        50,
        501,
        "2003-08-08",
        ["60-minutes-or-less", "main-dish", "chicken", "indian"],
        [480.0, 28.0, 10.0, 35.0, 75.0, 30.0, 8.0],
        ["brown chicken", "add spices and onion", "simmer with coconut milk"],
        "mild and creamy",
        ["chicken", "onion", "garlic", "curry powder", "coconut milk", "rice"],
    ),
    (
        113,
        "chocolate chip cookies",
        25,
        502,
        "2004-12-24",
        ["30-minutes-or-less", "desserts", "cookies-and-brownies"],
        [190.0, 15.0, 70.0, 5.0, 4.0, 30.0, 8.0],
        ["cream butter and sugar", "add flour and chips", "bake"],
        "chewy centres",
        ["flour", "butter", "sugar", "eggs", "chocolate chips", "baking soda"],
    ),
    (
        114,
        "fluffy pancakes",
        20,
        503,
        "2005-02-14",
        ["30-minutes-or-less", "breakfast"],
        [250.0, 12.0, 20.0, 15.0, 12.0, 20.0, 12.0],
        ["whisk dry ingredients", "add milk and eggs", "cook on griddle"],
        "sunday breakfast",
        ["flour", "milk", "eggs", "sugar", "baking powder", "butter"],
    ),
    (
        115,
        "chunky guacamole",
        10,
        504,
        "2006-09-16",
        ["15-minutes-or-less", "no-cook", "dips", "mexican"],
        [160.0, 22.0, 3.0, 12.0, 4.0, 10.0, 3.0],
        ["mash avocados", "stir in the rest"],
        "party favourite",
        ["avocado", "onion", "tomatoes", "lime juice", "salt"],
    ),
    (
        116,
        "caesar salad",
        15,
        505,
        "2007-06-06",
        ["15-minutes-or-less", "salads"],
        [300.0, 40.0, 3.0, 25.0, 15.0, 25.0, 3.0],
        ["tear lettuce", "make dressing", "toss with croutons"],
        "classic dressing",
        ["romaine lettuce", "parmesan cheese", "croutons", "olive oil", "lemon juice", "garlic"],
    ),
    (
        117,
        "roasted vegetables",
        45,
        506,
        "2008-08-18",
        ["60-minutes-or-less", "side-dishes", "vegetarian"],
        [140.0, 12.0, 15.0, 10.0, 6.0, 8.0, 6.0],
        ["chop vegetables", "toss with oil", "roast"],
        "easy side",
        ["carrots", "potatoes", "onion", "olive oil", "salt", "black pepper"],
    ),
    (
        118,
        "banana bread",
        70,
        507,
        "2009-03-03",
        ["breads", "baking", "desserts"],
        [260.0, 14.0, 80.0, 10.0, 8.0, 25.0, 13.0],
        ["mash bananas", "mix batter", "bake"],
        "uses overripe bananas",
        ["banana", "flour", "sugar", "eggs", "butter", "baking soda"],
    ),
    (
        119,
        "triple cheese lasagna",
        120,
        508,
        "2010-05-05",
        ["main-dish", "pasta", "italian"],
        [4520.3, 410.0, 95.0, 380.0, 520.0, 690.0, 110.0],
        ["make meat sauce", "layer pasta and cheese", "bake"],
        "feeds a crowd",
        [
            "lasagna noodles",
            "ground beef",
            "tomatoes",
            "mozzarella cheese",
            "ricotta cheese",
            "parmesan cheese",
            "garlic",
        ],
    ),
    (
        120,
        "scrambled eggs",
        10,
        509,
        "2011-11-11",
        ["15-minutes-or-less", "breakfast", "eggs"],
        [200.0, 22.0, 1.0, 9.0, 26.0, 25.0, 0.0],
        ["whisk eggs", "cook gently in butter"],
        "soft and creamy",
        ["eggs", "butter", "milk", "salt"],
    ),
]

# user_id -> list of (recipe_id, rating). rating 0 = "no stars given" in Food.com.
USER_RATINGS = {
    1001: [(101, 5), (102, 4), (112, 5), (113, 4), (114, 5), (120, 3)],
    1002: [(101, 4), (101, 4), (109, 5), (117, 4), (119, 5)],
    1003: [(103, 5), (104, 0), (115, 5), (116, 4), (113, 5)],
    1004: [(105, 3), (112, 4), (117, 5), (118, 4)],
    1005: [(106, 2), (107, 4), (118, 5), (114, 4), (101, 5)],
    1006: [(108, 4), (112, 0), (120, 5), (102, 5)],
    1007: [(110, 3), (104, 5), (115, 4), (103, 4)],
    1008: [(119, 4), (101, 3), (109, 4), (116, 5), (102, 2)],
    1009: [(113, 5), (118, 0), (114, 5), (120, 4)],
    1010: [(105, 1), (117, 3), (107, 5)],
    1011: [(116, 4), (115, 5), (109, 5), (112, 3)],
    1012: [(119, 0), (101, 5), (113, 4)],
    1013: [(104, 4), (103, 5), (110, 2), (108, 3)],
    1014: [(102, 3), (114, 4), (118, 5), (120, 4), (106, 1)],
    1015: [(112, 5)],
}

REVIEW_BY_RATING = {
    0: "Have not made this yet but it looks great, saving it.",
    1: "Did not work for us at all.",
    2: "It was okay, needed a lot more seasoning.",
    3: "Decent, would make again with changes.",
    4: "Very good, the family liked it.",
    5: "Excellent! Perfect as written.",
}
MISSING_REVIEW = (1011, 116)


def build_recipes() -> pd.DataFrame:
    rows = []
    for rid, name, minutes, contrib, submitted, tags, nutr, steps, desc, ingr in RECIPES:
        rows.append(
            {
                "name": name,
                "id": rid,
                "minutes": minutes,
                "contributor_id": contrib,
                "submitted": submitted,
                "tags": repr(tags),
                "nutrition": "[" + ", ".join(str(v) for v in nutr) + "]",
                "n_steps": len(steps),
                "steps": repr(steps),
                "description": desc,
                "ingredients": repr(ingr),
                "n_ingredients": len(ingr),
            }
        )
    return pd.DataFrame(rows)


def build_interactions() -> pd.DataFrame:
    rows = []
    day = date(2002, 3, 1)
    seen: dict[tuple[int, int], date] = {}
    for uid, pairs in USER_RATINGS.items():
        for rid, rating in pairs:
            key = (uid, rid)
            if key in seen:  # exact duplicate row: same date too
                d = seen[key]
            else:
                day += timedelta(days=97)
                d = seen[key] = day
            review = None if key == MISSING_REVIEW else REVIEW_BY_RATING[rating]
            rows.append(
                {
                    "user_id": uid,
                    "recipe_id": rid,
                    "date": d.isoformat(),
                    "rating": rating,
                    "review": review,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    build_recipes().to_csv(HERE / "RAW_recipes.csv", index=False)
    build_interactions().to_csv(HERE / "RAW_interactions.csv", index=False)


if __name__ == "__main__":
    main()
