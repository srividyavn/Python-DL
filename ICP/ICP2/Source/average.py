num_plants = int(input())
heights = [int(x) for x in input().split()]
heights_list = list(set(heights))
print(sum(heights_list) / float(len(heights_list)))