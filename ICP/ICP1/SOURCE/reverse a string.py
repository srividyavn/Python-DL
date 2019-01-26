fname = input("Input your First Name : ")
lname = input("Input your Last Name : ")

str=""
for i in fname:
    str = i+str

str1=""
for j in lname:
    str1 = j+str1

print("hello",str1,str)

