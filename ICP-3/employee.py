class Employee:
    
    number_of_employees = 0
    total_salary = 0
    
    def __init__(self, name, age, salary, department):
        self.name = name
        self.age = age
        self.salary = salary
        self.department = department
        
        self.__class__.number_of_employees += 1
        self.__class__.total_salary += self.salary

    def getAverageSalary():
        avg = Employee.total_salary / Employee.number_of_employees
        return round(avg, 2)

class FulltimeEmployee(Employee):
    
    def __init__(self, name, age, salary, department):
        Employee.__init__(self, name, age, salary, department)

if __name__ == "__main__":
    
    print("--------Employee Database---------\n")
    
while True:
    bool = input("\n Do you want to enter employee details (yes/no): ")
    if bool == "yes":
        name = input("Enter Employee name: ")
        age = int(input("Enter Employee age: "))
        salary = int(input("Enter Employee salary: "))
        department = input("Enter Employee Department: ")
        
        Employee(name, age, salary, department)
    
    else:
        print("Total Number of Employees: " + str(Employee.number_of_employees))
        print("Final Average Salary of Employees: " + str(Employee.getAverageSalary()))
        break