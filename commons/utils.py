import pandas as pd
import random
import re
from typing import List, Tuple

def field_string_to_list(string:str) -> List[str]:
    pattern = r'"(.*?)"'
    matches = re.findall(pattern, string)

    matches_list = [pd.NA if m == 'NA' else m for m in matches]

    return matches_list

def field_time_to_string(string:str) -> str:
    pattern = r'PT(?:([0-9]+)H)?(?:([0-9]+)M)?(?:([0-9]+)S)?'
    matches = re.search(pattern, string).groups()

    time_string = 'NA'
    if matches[0] and matches[1]:
        time_string = f'{int(matches[0]):02d}:{int(matches[1]):02d}'
    elif matches[0]:
        time_string = f'{int(matches[0]):02d}:00'
    elif matches[1]:
        time_string = f'00:{int(matches[1]):02d}'

    return time_string

def dictionary_time_to_string(times_dictionary:dict[str, str]) -> str:
    pattern = r'([0-9]+ (?:mins?|hrs?))(?: and ([0-9]+ mins))?$'
    time_minutes = 0
    time_hours = 0

    for time in times_dictionary.values():
        matches = re.search(pattern, time)
        if matches:
            for g in matches.groups():
                if not g: continue

                gs = g.split(' ')
                if gs[1][-2:] == 'ns' or gs[-1][-2:] == 'in':
                    time_minutes += int(gs[0])
                else:
                    time_hours += int(gs[0])

    time_string = 'NA'
    if time_minutes or time_hours:
        time_hours += time_minutes // 60
        time_minutes %= 60
        time_string = f'{int(time_hours):02d}:{int(time_minutes):02d}'

    return time_string

def field_ingredients_quantities(ingredients:List[str], quantities_raw:List[str]) -> List[str]:
    # quantities = []
    # seen = set()
    # ingredients_num = len(ingredients)
    # quantities_num = len(quantities_raw)

    # i = 0
    # while i < ingredients_num:
    #     ingredient = ingredients[i]
    #     if ingredient in seen:
    #         del ingredients[i]
    #         ingredients_num -= 1
    #         continue
    #     seen.add(ingredient)

    #     ingredient = re.sub(r'[\s\(\)\*]+', r'.*', ingredient)

    #     quantity_raw = quantities_raw[i]
    #     quantity_raw = re.sub(r'[\s,\.]+', ' ', quantity_raw)
        
    #     match = re.search(ingredient, quantity_raw)

    #     if match:
    #         quantity = quantity_raw[:match.start()-1] if match else 'NA'
    #     else:
    #         match = re.search(r'salt|pepper|sugar', quantity_raw)
    #         if match:
    #             new_ingredient = quantity_raw[match.start():match.end()+1]
    #             quantity = quantity_raw[:match.start()-1]
    #             ingredients.insert(i, new_ingredient)
    #             ingredients_num += 1
    #         else:
    #             quantity = 'NA'

    #     quantity = 'NA' if quantity.isspace() else quantity
    #     quantities.append(quantity)
    #     i += 1

    # while i < quantities_num:
    #     quantity_raw = quantities_raw[i]
    #     quantity = 'NA'
    #     match = re.search(r'salt|pepper|sugar', quantity_raw)

    #     if match:
    #         new_ingredient = quantity_raw[match.start():match.end()+1]
    #         quantity = quantity_raw[:match.start()-1]
    #         ingredients.insert(i, new_ingredient)
    #         quantity = 'NA' if quantity.isspace() else quantity

    #     quantities.append(quantity)
    #     i += 1

    quantities = []

    for ingredient, quantity_raw in zip(ingredients, quantities_raw):
        quantity_raw = re.sub(r'[\s,\.]+', ' ', quantity_raw)
        try:
            pattern = re.sub(r'[\s\(\)\*]+', r'.*', ingredient)
            match = re.search(pattern, quantity_raw)
            quantity = quantity_raw[:match.start()-1] if match else 'NA'
            quantity = 'NA' if quantity.isspace() else quantity
        except re.error:
            quantity = 'NA'
        quantities.append(quantity)

    return quantities

def dataframe_information(dataframe:pd.DataFrame) -> str:
    information = \
    f'Shape: {dataframe.shape}\n' + \
    f'Column headers: {dataframe.columns.values.tolist()}\n' + \
    f'Num. of null rows: {dataframe.isnull().any(axis=1).sum()}\n' + \
    f'Num. of null features (by column):\n{dataframe.isnull().sum(axis=0)}\n' + \
    f'First 5 rows:\n{dataframe.head(5)}\n' + \
    f'Last 5 rows:\n{dataframe.tail(5)}\n' + \
    f'Random sample:\n{dataframe.iloc[random.randint(0, len(dataframe)-1)]}'

    return information
