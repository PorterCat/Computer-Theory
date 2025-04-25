#include <iostream>
#include <algorithm> 

/*
Задача. Имеется склад, на котором есть некоторый ассортимент товаров. Запас каждого товара считается неограниченным. Товары имеют две характеристики: mi – масса, ci – стоимость; 
Необходимо выбрать набор товаров так, что бы его суммарная масса не превосходила заранее фиксированную массу М(т.е. Σmi  ≤ M), и стоимость набора была как можно больше (Σci→max).
*/

struct Good 
{
    int mass = 0;
    int cost = 0;
    int usedTimes = 0;
};

int Solution(Good* goods, int k, const int m)
{
    int fs[m + 1] {}; 
    int maxs[k] {};

    for(int i = 0; i <= m; ++i)
    {
        for(int j = 0; j < k; ++j)
        {
            if(i - goods[j].mass >= 0)
                maxs[j] = fs[i - goods[j].mass] + goods[j].cost;
        }

        int* max_it = std::max_element(maxs, maxs + k);
        fs[i] = *max_it;
    }

    int weight = m;
    for(int i = 0; i < k; ++i)
        while(weight >= goods[i].mass && fs[weight] == fs[weight - goods[i].mass] + goods[i].cost)
        {
            goods[i].usedTimes += 1;
            weight -= goods[i].mass;
        }

    return fs[m];
}

int main()
{
    Good goods[3] = {{3, 8}, {5, 14}, {8, 23}};
    const int M = 19;

    size_t k = sizeof(goods) / sizeof(Good);

    std::cout << "Backpack capacity: " << M << "kg" << std::endl;
    std::cout << "Max price: " << Solution(goods, k, M) << "$" << std::endl;

    for(int i = 0; i < k; ++i)
        std::cout << "Item " << i << " (" << goods[i].mass << " kg/" << goods[i].cost <<  "$) : " << goods[i].usedTimes << std::endl;
}