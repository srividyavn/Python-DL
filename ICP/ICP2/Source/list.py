
lis = []
for i in range(4):
    lis.append(input("enter element"))
print(lis)
print(lis.pop())
print(lis.pop())
print(lis)



from collections import deque
queue = deque(["himalayas", "1", "beautiful", "2"])
print(queue)
queue.append("Akbar")
print(queue)
queue.append("Birbal")
print(queue)
print(queue.popleft())
print(queue.popleft())
print(queue)