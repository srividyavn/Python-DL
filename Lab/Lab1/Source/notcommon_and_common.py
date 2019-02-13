#To find the common student in two classes and to print common and non-common student from the class..

#declaring empty lists
Python = []
Web = []

print("Enter students of python class ******** Press a to exit *******: ")
while(True): #loop to append the data into python class
    temp = input("Enter name: ")

    if(temp =='a'):
        break
    else:
        Python.append(temp)

print("Enter students of Web class ******** Press a to exit *******: ")
while (True): #loop to append the data into web class
    temp = input("Enter name: ")

    if (temp == 'a'):
        break
    else:
        Web.append(temp)

#to print the common students in both class
print("students enroled in both the classes are: \n")
for a in Python: #loop for finding common students
    if a in Web:
        print(a)


#to print the non-common students in both class
print("students who are not common in both the classes: \n")
for b in Python: #loop for finding non-common students
    if b not in Web:
        print(b)
for c in Web:
    if c not in Python:
        print(c)
