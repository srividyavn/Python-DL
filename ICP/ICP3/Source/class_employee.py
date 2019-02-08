# Employee class
class Employee:
    countEmployees = 0
    salaries = []

    #Default constructor function
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department

        #appending salaries to the list
        Employee.salaries.append(self.salary)
        Employee.countEmployees = self.countEmployees + 1

    #Average salary of all employess
    def avgsalary(self, salaries):
        length = len(salaries)
        totalsalary = 0
        for salary in salaries:
            totalsalary = totalsalary + salary
        print("Average Salary = ", totalsalary/length)


#Full time Employee class
class FulltimeEmployee(Employee):
    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)

    def testing(self):
        print("subclass test")

Employee1 = Employee("sri", "Vidya", 100000,"BigData")

Employee2 = Employee("Bill", "Games", 2000986,"Java")
Employee3 = FulltimeEmployee("Mark", "Thames", 2000236,"Python")
Employee3.avg
Employee4 = FulltimeEmployee("Hiliary", "Jain", 590836,"C")
Employee5 = Employee("Kate", "chefs", 1734799,"Mainframe")
print(Employee1.name)
print(Employee2.name)
print(Employee3.name)
print(Employee4.name)
print(Employee5.name)

# Access data member using FulltimeEmployee class
print("Number of Employees: ", FulltimeEmployee.countEmployees)