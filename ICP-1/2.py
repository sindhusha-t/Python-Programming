print("---------Reversing a string after deleting atleast 2 characters----------\n")
string1 = input("Enter any string:")

# Deleting 2 characters
s_list = string1[2:]

# Reversing the string
r_string = s_list[::-1]
print(r_string)

print()
print("---------Doing Arithmetic Operations----------\n")

input1 = input("Enter first Number:")
input2 = input("Enter Second Number:")

num1 = int(input1)
num2 = int(input2)

print("%d + %d = %d" %(num1, num2, num1+num2))
print("%d - %d = %d" %(num1, num2, num1-num2))
print("%d * %d = %d" %(num1, num2, num1*num2))
print("%d / %d = %d" %(num1, num2, num1/num2))
print("%d // %d = %d" %(num1, num2, num1//num2))