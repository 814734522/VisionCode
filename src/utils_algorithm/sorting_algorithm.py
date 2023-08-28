"""
此模块由于存放排序算法

@author: lfc
"""


def quick_sort(lists, i, j):
    if i >= j:
        return lists
    pivot = lists[i]
    low = i
    high = j
    while i < j:
        while i < j and lists[j][1] >= pivot[1]:
            j -= 1
        lists[i] = lists[j]
        while i < j and lists[i][1] <= pivot[1]:
            i += 1
        lists[j] = lists[i]
    lists[j] = pivot
    quick_sort(lists, low, i - 1)
    quick_sort(lists, i + 1, high)
    return lists
