def process_output(output):
    # Словарь для хранения данных
    table = {
        "Без переполнения": {},
        "С одним переполнением": {},
        "С двумя переполнениями": {}
    }

    # Обрабатываем каждую строку вывода
    for line in output.strip().split("\n"):
        parts = line.split()
        razryad = int(parts[1])  # Разрядность
        overflow_types = parts[2:]  # Типы переполнений

        # Обновляем счетчики для каждого типа переполнения
        for overflow_type in overflow_types:
            # Убедимся, что тип переполнения корректно обрабатывается
            if overflow_type == "Без":
                overflow_type = "Без переполнения"
            elif overflow_type == "С":
                continue  # Пропускаем "С" (оно не является полным типом)
            elif overflow_type == "одним":
                overflow_type = "С одним переполнением"
            elif overflow_type == "двумя":
                overflow_type = "С двумя переполнениями"
            elif overflow_type == "переполнения" or overflow_type == "переполнением" or overflow_type == "переполнениями":
                continue  # Пропускаем "переполнения", "переполнением" и "переполнениями" (они не являются полными типами)

            if razryad not in table[overflow_type]:
                table[overflow_type][razryad] = 0
            table[overflow_type][razryad] += 1

    return table


def print_table(table):
    # Выводим заголовок таблицы
    header = "Разрядность | Без переполнения | С одним переполнением | С двумя переполнениями | Итого"
    print(header)
    print("-" * len(header))

    # Выводим данные для каждого разряда
    for razryad in sorted(set(table["Без переполнения"].keys()) | set(table["С одним переполнением"].keys()) | set(table["С двумя переполнениями"].keys())):
        no_overflow = table["Без переполнения"].get(razryad, 0)
        one_overflow = table["С одним переполнением"].get(razryad, 0)
        two_overflows = table["С двумя переполнениями"].get(razryad, 0)
        total = no_overflow + one_overflow + two_overflows  # Итого для каждого разряда
        print(f"{razryad:^11} | {no_overflow:^16} | {one_overflow:^21} | {two_overflows:^22} | {total:^5}")

    # Выводим итоговую строку для каждого типа переполнения
    total_no_overflow = sum(table["Без переполнения"].values())
    total_one_overflow = sum(table["С одним переполнением"].values())
    total_two_overflows = sum(table["С двумя переполнениями"].values())
    total_all = total_no_overflow + total_one_overflow + total_two_overflows
    print(f"{'Итого':^11} | {total_no_overflow:^16} | {total_one_overflow:^21} | {total_two_overflows:^22} | {total_all:^5}")


# Пример вывода из вашего кода
output = """
RAZRYAD: 4 Без переполнения
RAZRYAD: 4 Без переполнения
RAZRYAD: 4 С одним переполнением
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 С одним переполнением
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С одним переполнением
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С одним переполнением
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С одним переполнением
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 С двумя переполнениями
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С одним переполнением
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С двумя переполнениями
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С одним переполнением
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 Без переполнения
RAZRYAD: 2 С одним переполнением
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 Без переполнения
RAZRYAD: 1 С двумя переполнениями
"""

# Обрабатываем вывод
table = process_output(output)

# Выводим таблицу
print_table(table)
