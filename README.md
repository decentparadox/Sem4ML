

# Assignment 1 Documentation

## Teammate's GitHub

Visit 

[Krishna varma's GitHub](https://github.com/teammateusername)
[Kushal's GitHub](https://github.com/teammateusername)


for their contributions to the project.

---

## Algorithm Explanation
TODO:
 Set A  Q1, 
Set c to 0
Set a as the given list [2, 7, 4, 1, 3, 6]
Create an empty list b

For each element a[i] in a:
    For each element a[j] in a starting from index i+1:
        If the sum of a[i] and a[j] equals 10:
            Increment c by 1
            Append a tuple (a[i], a[j]) to list b
        Else:
            Do nothing

Print the value of c
Print the list b
Output: it prints the pairs which gives the output 10
 Set A  Q2, 
 take user input as and store it in string variable
 For each element x in string:
   covert into int
print the range from max to min
 
Set A  Q3,
FUNCTION matrixmult(matrix, matrix1)
    IF length of matrix is not equal to length of matrix1
        RAISE ValueError("Number of rows in the first matrix must equal the number of rows in the second matrix.")

    result = [] # Initialize an empty list to store the result of matrix multiplication

    FOR each row index i in the range of length of matrix
        row = [] # Initialize an empty list to store each row of the resulting matrix
        FOR each column index j in the range of length of the first row of matrix1
            sum = 0 # Initialize sum to 0 for the current element of the resulting matrix
            FOR each index k in the range of length of matrix1
                sum += matrix[i][k] * matrix1[k][j] # Perform the matrix multiplication operation
            row.append(sum) # Append the sum to the current row
        result.append(row) # Append the row to the result list
    RETURN result # Return the resulting matrix

function powermatrix(matrix, m)
    if m equals 1
        return matrix
   else
      return matrixmult(matrix, powermatrix(matrix, m-1))

matrix = [] # Initialize an empty list to store the matrix elements

# Input elements for the first row of the matrix
string = input("Enter numbers in first row of the matrix").split()
matrix.append(string)

no_rows = length of string # Get the number of rows in the matrix

for each _ in the range of no_rows - 1
    # Input elements for the next row of the matrix
    string = input("Enter numbers in first row of the matrix").split()
    matrix.append(string)

for each row x in matrix
    FOR each element y in row x
        y = convert y to an integer

m = convert input("Enter a number m") to an integer

# Calculate the result using the powermatrix function
result = powermatrix(matrix, m)

# Print the result matrix and the number of rows in the original matrix
print(result)
print(length of matrix)

Output:Enter numbers in first row of the matrix:
Enter numbers in next row of the matrix
Enter numbers in next row of the matrix
Enter a number m
it gives result 
of the multiplication
 Set A  Q4,
x = input("Enter the string:") # Take user input for a string

print(x) # Print the input string

count = {} # Initialize an empty dictionary to store character counts

for each character i in x: # Iterate through each character in the input string
    IF i is already a key in count:
        Increment the value associated with i in count by 1
    ELSE:
        Set the value associated with i in count to 1

result = CHARACTER with maximum value in count # Find the character with the highest occurrence count
count = VALUE associated with result in count # Get the count of the maximally occurring character

print("Maximally occurring character:", result, "and the count is", count) # Print the maximally occurring character and its count



### Set B
#### Pseudocode 

Set B Q1
```plaintext
    Input: Prompt the user to enter a string.
    Initialize Variables:
        Set vowels to the string "aeiou".
        Set vowel_count and character_count to 0.
    Iterate Through Characters:
        For each character c in the input string:
            Check if c is an alphabetic character.
                If true:
                    Increment character_count by 1.
                    Check if the lowercase version of c is in the set of vowels.
                        If true, increment vowel_count by 1.
    Output Results: Display the count of vowels and consonants:
        Print "Vowels: " followed by vowel_count.
        Print "Consonants: " followed by (character_count - vowel_count).
```

Set B Q2
```plaintext 
        Input Matrices A and B:
        Ask for the size and elements of matrix A.
        Ask for the size and elements of matrix B.

    Check Compatibility:
        Make sure the number of columns in A equals the number of rows in B.
        If not, show an error message and stop.

    Multiply Matrices:
        Create a matrix for the result filled with zeros.
        Multiply corresponding elements of A and B, summing them up for each position in the result matrix.

    Output:
        If there's an error, show an error message.
        Otherwise, display matrices A, B, and the multiplied result AB.
```

Set B Q3
```plaintext
    Input Lists:
        Prompt the user to enter elements for the first list separated by spaces.
        Prompt the user to enter elements for the second list separated by spaces.

    Convert Input to Integer Lists:
        Convert the input strings to integer lists for both lists.

    Find Common Elements:
        Use sets to find the common elements between the two lists.

    Calculate Common Elements Count:
        Calculate the number of common elements.

    Output Result:
        Print the count of common elements.
```

Set B Q4

```plaintext
    Input Matrix:
        Prompt the user to enter the number of rows and columns for the matrix.
        Use the input_matrix function to input the elements for the matrix.

    Transpose Matrix:
        Use the transpose_matrix function to find the transpose of the input matrix.
            Create a new matrix where the rows become columns and vice versa.

    Output Results:
        Print the original matrix.
        Print the transposed matrix.

```
