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