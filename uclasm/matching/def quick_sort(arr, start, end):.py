 

def quick_sort(arr, start, end):
    if start >= end:
        return
    
    print('------ Quick sorting from index: ', start, ' to ', end, ' -----')
    print('Array segment before partitioning: ', end=' ')
    for i in range(start, end + 1):
        print(arr[i], end=' ')
    print(', Pivot: ', arr[end])
    
    k = partition(arr, start, end)
    
    
    print('Array segment after partitioning: ', end=' ')
    for i in range(start, end + 1):
        print(arr[i], end=' ')
    print('\n')
    
    
    quick_sort(arr, start, k-1)
    quick_sort(arr, k+1, end)

def partition(arr, low, high):
    pivot = arr[high]

    i = low

    for j in range(low,high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]

    return i

arr = [11, 12, 1, 9, 6, 5, 4, 7]
quick_sort(arr, 0, len(arr)-1)
print('The final sorted array', arr)