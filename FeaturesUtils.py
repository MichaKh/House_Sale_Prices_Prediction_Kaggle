num_stories_dict = {'1Story': '1', '2Story': '2', 'SFoyer': '2', 'SLvl': '3', '1.5Fin': '1.5', '1.5Unf': '1.5',
                    '2.5Fin': '2.5', '2.5Unf': '2.5'}

proximity_condition_arterial_location = ['Artery', 'RRNn', 'RRAn', 'RRNe', 'RRAe']
proximity_condition_off_site_location = ['PosN', 'PosA']

month_to_season = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}


def get_num_of_stories(x):
    if x in num_stories_dict:
        return num_stories_dict[x]
    else:
        return 'Unknown'


def get_house_age(x):
    age = 1
    try:
        age = 2013 - x
    except Exception:
        pass
    return age


def was_the_house_renovated(construction_year, renovation_year):
    was_renovated = False
    if construction_year < renovation_year:
        if 2013 - renovation_year <= 10:
            was_renovated = True
        else:
            was_renovated = False

    return was_renovated


def has_alley_access(x):
    has_alley = False
    if x != 'NA':
        has_alley = True
    return has_alley


def adjacent_to_artery(first_condition, second_condition):
    adjacent_to_arterial_location = False
    if first_condition in proximity_condition_arterial_location or second_condition in proximity_condition_arterial_location:
        adjacent_to_arterial_location = True
    return adjacent_to_arterial_location


def adjacent_to_off_sites(first_condition, second_condition):
    adjacent_to_off_sites_location = False
    if first_condition in proximity_condition_off_site_location or second_condition in proximity_condition_off_site_location:
        adjacent_to_off_sites_location = True
    return adjacent_to_off_sites_location


def has_basement(basement_cond):
    has_bstm = False
    if basement_cond != 'NA':
        has_bstm = True
    return has_bstm


def has_house_feature(house_feature):
    has_feature = False
    if house_feature != 'NA':
        has_feature = True
    return has_feature


def sale_season(x):
    if x in month_to_season:
        return month_to_season[x]
    else:
        return 'Unknown'


def sold_in_last_3years(sold_year):
    sold_in_last_years = False
    try:
        if 2013 - sold_year <= 3:
            sold_in_last_years = True
    except Exception:
        pass
    return sold_in_last_years




