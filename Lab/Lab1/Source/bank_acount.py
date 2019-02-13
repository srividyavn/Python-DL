#computeing the net amount of a bank account

c = []
total_amount = 0

#taking the input from user
print('enter input')

#loop through all the input records
while True:
    c = input()
    if c:

        #splitting each record add performing the operation accordingly

        c = c.split(" ")
        if c[0] == "deposit":
            total_amount = total_amount + int(c[1])
        elif c[0] == "withdraw":
            total_amount = total_amount - int(c[1])
    else:
        break
print("total amount is ", total_amount)





