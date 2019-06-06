

def string_alternative(input_str):

    #Using List Slicing
    print(input_str[::2])

    #Using For Loops
    output_str = [input_str[index] for index in range(0,len(input_str), 2)]
    
    print(''.join(output_str))
    

def main():
    print("\--------PRINT ALTERNATE CHARS IN STRINGS----------n")

    input_str = input("Enter a sentence: ")
    string_alternative(input_str)

if __name__ == "__main__":
    main()
