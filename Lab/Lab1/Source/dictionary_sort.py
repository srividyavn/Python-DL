#Converting list of tuples into Dictionary with keys and values and convert into sorted order.

input = [('John', ('Physics', 80)), ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)),
         ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]

#empty dictionary is created
result_dict = {}

#Iterate for each and every item in the list
for item in input:
    res_tuple = item
    if res_tuple[0] in result_dict.keys():
        result_dict[res_tuple[0]].append(res_tuple[1]) #if exists then add current value to key
    else:
        result_dict[res_tuple[0]] = [res_tuple[1]] #if not then add new key and valur pair to dictionary.

#print the output of dictionary in sorted order
for key, value in result_dict.items():
    print(key, sorted(value))
