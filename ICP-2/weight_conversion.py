print("--------WEIGHT CONVERSION--------\n")

list_str = input("Enter list of weights in pounds separated by \"space\": ")
pounds = list_str.split()

#Using Loops
kgs = []
for pound in pounds:
    kgs.append(str(  round(  int(pound)/2.205 , 2) ))

print(','.join(kgs))


#Using List comprehensions
kgs = [str(  round(  int(pound)/2.205 , 2) ) for pound in pounds]
print(','.join(kgs))


print("--------USING SINGLE FOR LOOP---------\n")

out_list = []
num_of_ele = int(input("Enter Number of elements: "))
for i in range(0, num_of_ele):
    value = int(input("Enter Element>>: "))
    out_list.append(round(  int(value)/2.205 , 2))

print(out_list)