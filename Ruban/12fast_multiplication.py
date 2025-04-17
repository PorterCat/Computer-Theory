multiplication_counts = {}

def increment_count(n_half, overflow_type):
    if n_half not in multiplication_counts:
        multiplication_counts[n_half] = {
            "Без переполнения": 0,
            "С одним переполнением": 0,
            "С двумя переполнениями": 0,
        }
    multiplication_counts[n_half][overflow_type] += 1

def handle_overflow(value, n_half):
    if len(value) > n_half:
        return "1", value[1:]
    else:
        return "0", value


def check_overflow(a, b, m):
    if len(a) == m + 1 and len(b) == m + 1:
        return "С двумя переполнениями"
    elif len(a) == m + 1 or len(b) == m + 1:
        return "С одним переполнением"
    return "Без переполнения"


def karatsuba_with_overflow(x, y):
    x_str = str(x)
    y_str = str(y)

    n = max(len(x_str), len(y_str))
    n_half = n // 2

    if n == 1:
        overflow_type = check_overflow(x_str, y_str, 1)
        return int(x_str) * int(y_str)

    a = x_str[:n_half]
    b = x_str[n_half:]
    c = y_str[:n_half]
    d = y_str[n_half:]

    a = a if a else "0"
    b = b if b else "0"
    c = c if c else "0"
    d = d if d else "0"

    a_plus_b = str(int(a) + int(b))
    c_plus_d = str(int(c) + int(d))

    # print(
    #     f"U = (a + b) * (c + d) = ({int(a)} + {int(b)}) * ({int(c)} + {int(b)})= {a_plus_b} * {c_plus_d}"
    # )
    # print(f"V = (a * c) = {a} * {c}")
    # print(f"W = (b * d) = {b} * {d}")

    type_p1 = check_overflow(a, c, n_half)
    type_p2 = check_overflow(b, d, n_half)
    type_p3 = check_overflow(a_plus_b, c_plus_d, n_half)
    increment_count(n_half, type_p1)
    increment_count(n_half, type_p2)
    increment_count(n_half, type_p3)
    # print("RAZRYAD:", n_half, a_plus_b, c_plus_d, type_p3)
    # print("RAZRYAD:", n_half, a, c, type_p1)
    # print("RAZRYAD:", n_half, b, d, type_p2)
    # print("RAZRYAD:", n_half, type_p1)
    # print("RAZRYAD:", n_half, type_p2)
    # print("RAZRYAD:", n_half, type_p3)

    if len(a_plus_b) > n_half or len(c_plus_d) > n_half:
        a1, a2 = handle_overflow(a_plus_b, n_half)
        c1, c2 = handle_overflow(c_plus_d, n_half)

        if n_half > 1:
            a_plus_b = a2[0 : (n_half // 2)] + a2[n_half // 2 :]
            c_plus_d = c2[0 : (n_half // 2)] + c2[n_half // 2 :]
            u = karatsuba_with_overflow(a_plus_b, c_plus_d)

        u = (
            (int(a1) * int(c1)) * (10 ** (2 * n_half))
            + (int(a1) * int(c2) + int(a2) * int(c1)) * (10**n_half)
            + (int(a2) * int(c2))
        )
    else:
        u = karatsuba_with_overflow(a_plus_b, c_plus_d)

    v = karatsuba_with_overflow(a, c)
    w = karatsuba_with_overflow(b, d)

    return v * (10 ** (2 * n_half)) + (u - v - w) * (10**n_half) + w


# x = 6789
# x = 3871
# y = 9211
# x = 8329  
# y = 5631
# x = "5055206520"
# y = "8168894301"
# x = "80634290"
# y = "97253579"
# x = "54093116666411635101896745211080"
# y = "97334494566701330872518141737034"
x = "9733449456670133087251814173703454093116666411635101896745211080"
y = "5409311666641163510189674521108097334494566701330872518141737034"
# x = "54093116666411635101896745211080973344945667013308725181417370349733449456670133087251814173703454093116666411635101896745211080"
# y = "97334494566701330872518141737034540931166664116351018967452110805409311666641163510189674521108097334494566701330872518141737034"
# x = "9733449456670133087251814173703454093116666411635101896745211080540931166664116351018967452110809733449456670133087251814173703454093116666411635101896745211080973344945667013308725181417370349733449456670133087251814173703454093116666411635101896745211080"
# y = "5409311666641163510189674521108097334494566701330872518141737034973344945667013308725181417370345409311666641163510189674521108097334494566701330872518141737034540931166664116351018967452110805409311666641163510189674521108097334494566701330872518141737034"
# y = 5671
print(karatsuba_with_overflow(x, y))

print("Таблица умножений по разрядам и типам:")
print(
    f"{'Разряд':<10} {'Без переполнения':<20} {'С одним переполнением':<25} {'С двумя переполнениями':<25} {'Итого':<10}"
)

total_by_type = {
    "Без переполнения": 0,
    "С одним переполнением": 0,
    "С двумя переполнениями": 0,
}

for n_half, counts in sorted(multiplication_counts.items()):
    total_for_n_half = (
        counts["Без переполнения"]
        + counts["С одним переполнением"]
        + counts["С двумя переполнениями"]
    )
    print(
        f"{n_half:<10} {counts['Без переполнения']:<20} {counts['С одним переполнением']:<25} {counts['С двумя переполнениями']:<25} {total_for_n_half:<10}"
    )

    total_by_type["Без переполнения"] += counts["Без переполнения"]
    total_by_type["С одним переполнением"] += counts["С одним переполнением"]
    total_by_type["С двумя переполнениями"] += counts["С двумя переполнениями"]

total_overall = (
    total_by_type["Без переполнения"]
    + total_by_type["С одним переполнением"]
    + total_by_type["С двумя переполнениями"]
)
print(
    f"{'Итого':<10} {total_by_type['Без переполнения']:<20} {total_by_type['С одним переполнением']:<25} {total_by_type['С двумя переполнениями']:<25} {total_overall:<10}"
)
