import numpy as np


el_array = [
    np.genfromtxt(
        fname='global-electricity-consumption.csv',
        dtype=str,
        skip_header=1,
        delimiter=',',
        usecols=(0,)
    ),
    np.genfromtxt(
        fname='global-electricity-generation.csv',
        skip_header=1,
        delimiter=',',
        usecols=([i for i in range(1, 31)])
    ),
    np.genfromtxt(
        fname='global-electricity-consumption.csv',
        skip_header=1,
        delimiter=',',
        usecols=([i for i in range(1, 31)])
    )
]

max_name_len = max(len(country) for country in el_array[0])
max_num_len = max(
    len(f"{int(np.nanmax(el_array[1]))}.000"),
    len(f"{int(np.nanmax(el_array[2]))}.000")
)

# Часть 1

mean_gen_for_last_5 = np.mean(el_array[1][:, -5:], axis=1)
mean_con_for_last_5 = np.mean(el_array[2][:, -5:], axis=1)

local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(max_num_len, len("Сгенерировано"), len("Потреблено"))
print(
    "\nСреднее ежегодное производство и потребление электроэнергии за последние 5 лет.\n" +
    ("—" * (local_max_name_len + local_max_num_len + local_max_num_len + 14)) +
    f"\n{"Страна":<{local_max_name_len}}   |   {"Сгенерировано":>{local_max_num_len}}"
    f"   |   {"Потреблено":>{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + (local_max_num_len + 7) * 2))
)
for i in range(len(el_array[0])):
    print(
        f"{el_array[0][i]:<{local_max_name_len}}   |   "
        f"{round(mean_gen_for_last_5[i], 3):>{local_max_num_len}.3f}   |   "
        f"{round(mean_con_for_last_5[i], 3):>{local_max_num_len}.3f}"
    )
print("—" * (local_max_name_len + (local_max_num_len + 7) * 2))

# Часть 2

sum_con_by_years = np.nansum(el_array[2], axis=0)

local_max_year_len = 4
local_max_num_len = len("Мировое потребление электричества в млрд. кВт*ч")
print(
    "\nСуммарное (по всем странам) потребление электроэнергии за каждый год.\n" +
    ("—" * (local_max_year_len + local_max_num_len + 7)) +
    f"\n{"Год":<{local_max_year_len}}   |   {"Мировое потребление электричества в млрд. кВт*ч":<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_year_len + local_max_num_len + 7))
)
for i in range(len(sum_con_by_years)):
    print(
        f"{(i+1992):<{local_max_year_len}}   |   {round(sum_con_by_years[i], 3):>{local_max_num_len}.3f}"
    )
print("—" * (local_max_year_len + local_max_num_len + 7))

# Часть 3

max_gen_by_country = np.nanmax(el_array[1], axis=1)

local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = len("MAX электричества произведённого за год")
print(
    "\nМаксимальное количество электроэнергии, которое произвела одна страна за один год.\n" +
    ("—" * (local_max_name_len + local_max_num_len + 7)) +
    f"\n{"Страна":<{local_max_name_len}}   |   {"MAX электричества произведённого за год":<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + 7))
)
for i in range(len(el_array[0])):
    print(
        f"{el_array[0][i]:<{local_max_name_len}}   |   {round(max_gen_by_country[i], 3):>{local_max_num_len}.3f}"
    )
print("—" * (local_max_name_len + local_max_num_len + 7))

# Часть 4

country_gen_more_500: list[tuple] = []
for i in range(len(mean_gen_for_last_5)):
    country = el_array[0][i]
    mean = mean_gen_for_last_5[i]
    if mean > 500:
        country_gen_more_500.append((country, mean))

country_gen_more_500.sort(key=lambda x: -x[1])
local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(max_num_len, len("Произведено"))
print(
    "\nСписок стран, которые производят более 500 млрд. кВт*ч электроэнергии ежегодно в среднем за последние 5 лет.\n"
    + ("—" * (local_max_name_len + local_max_num_len + 7)) +
    f"\n{"Страна":<{local_max_name_len}}   |   {"Произведено":<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + 7))
)
for country, mean in country_gen_more_500:
    print(f"{country:<{local_max_name_len}}   |   {round(mean, 3):>{local_max_num_len}.3f}")
print("—" * (local_max_name_len + local_max_num_len + 7))

# Часть 5

country_con_more_p90: list[tuple] = []
p90 = np.percentile(a=mean_con_for_last_5, q=90)
local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(max_num_len, len("Энергопотребление"))
print(
    f"\nСтраны, входящие в 10% стран с наибольшим средним энергопотреблением за последние 5 лет, "
    f"при процентиле 90%, равному {round(p90, 3)}\n" +
    ("—" * (local_max_name_len + local_max_num_len + 7)) +
    f"\n{"Страна":<{local_max_name_len}}   |   {"Энергопотребление":<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + 7))
)
for i in range(len(mean_con_for_last_5)):
    if mean_gen_for_last_5[i] > p90:
        country_con_more_p90.append((el_array[0][i], mean_con_for_last_5[i]))
country_con_more_p90.sort(key=lambda x: -x[1])
for country, num in country_con_more_p90:
    print(
        f"{country:<{local_max_name_len}}   |   {round(num, 3):>{local_max_num_len}.3f}"
    )
print("—" * (local_max_name_len + local_max_num_len + 7))

# Часть 6

local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(max_num_len, len("Произведено в 1992 году"), len("Произведено в 2021 году"))
print(
    f"\nСписок стран, которые увеличили производство электроэнергии в 2021 году "
    f"по сравнению с 1992 годом более, чем в 10 раз.\n" +
    ("—" * (local_max_name_len + local_max_num_len + local_max_num_len + 14)) +
    f"\n{"Страна":<{local_max_name_len}}   |   {"Произведено в 1992 году":<{local_max_num_len}}"
    f"   |   {"Произведено в 2021 году":<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + local_max_num_len + 14))
)
for i in range(len(el_array[1])):
    if el_array[1][i][-1] > el_array[1][i][0] * 10:
        print(
            f"{el_array[0][i]:<{local_max_name_len}}   |   {el_array[1][i][0]:>{local_max_num_len}.3f}"
            f"   |   {el_array[1][i][-1]:>{local_max_num_len}.3f}"
        )
print("—" * (local_max_name_len + local_max_num_len + local_max_num_len + 14))

# Часть 7

sum_gen_all_years = np.nansum(el_array[1], axis=1)
sum_con_all_years = np.nansum(el_array[2], axis=1)

countries_filtered = []
for i in range(len(el_array[0])):
    if sum_con_all_years[i] > 100 and sum_gen_all_years[i] < sum_con_all_years[i]:
        countries_filtered.append((el_array[0][i], sum_con_all_years[i], sum_gen_all_years[i]))
countries_filtered.sort(key=lambda x: -x[1])
local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(
    max_num_len,
    len("Суммарное потребление"),
    len("Суммарное производство")
)
print(
    "\nСтраны, которые потратили в сумме за все годы больше 100 млрд. кВт*ч "
    "и произвели меньше, чем потратили:\n" +
    ("—" * (local_max_name_len + local_max_num_len * 2 + 14)) +
    f"\n{'Страна':<{local_max_name_len}}   |   {'Суммарное потребление':>{local_max_num_len}}"
    f"   |   {'Суммарное производство':>{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len * 2 + 14))
)

for country, total_con, total_gen in countries_filtered:
    print(
        f"{country:<{local_max_name_len}}   |   "
        f"{round(total_con, 3):>{local_max_num_len}.3f}   |   "
        f"{round(total_gen, 3):>{local_max_num_len}.3f}"
    )
print("—" * (local_max_name_len + (local_max_num_len + 7) * 2))

# Часть 8

consumption_2020 = el_array[2][:, -2]
max_consumption_idx = np.nanargmax(consumption_2020)
max_consumption_country = el_array[0][max_consumption_idx]
max_consumption_value = consumption_2020[max_consumption_idx]
local_max_name_len = max(max_name_len, len("Страна"))
local_max_num_len = max(max_num_len, len("Потребление в 2020 году"))
print(
    "\nСтрана с наибольшим потреблением электроэнергии в 2020 году:\n" +
    ("—" * (local_max_name_len + local_max_num_len + 7)) +
    f"\n{'Страна':<{local_max_name_len}}   |   {'Потребление в 2020 году':<{local_max_num_len}}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + 7)) +
    f"\n{max_consumption_country:<{local_max_name_len}}   |   "
    f"{round(max_consumption_value, 3):>{local_max_num_len}.3f}"
    f"\n" + ("—" * (local_max_name_len + local_max_num_len + 7))
)