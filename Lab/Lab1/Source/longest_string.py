#program to find the longest string and length

length = input("enter the string ")
current_longest = ""
longest_string = ""

#declaring a set
set_string = set()

#for loop to check each and every character in set
for i in range(0, len(length)):

    c = length[i]#to check if character already exists in set

    if c in set_string:#if it exists then empty the current set.
        current_longest = ""
        set_string.clear()

    current_longest = current_longest + c#if does not exist then add it yo current_longest
    set_string.add(c)

    if len(current_longest) > len(longest_string):#check the length of current string and previous string
        longest_string = current_longest
print("the longest substring is", longest_string)
print(" length is ", len(longest_string))#print the longest string and length