

# Assignment 1 Documentation

## Teammate's GitHub

Visit 

[Krishna varma's GitHub](https://github.com/teammateusername)
[Kushal's GitHub](https://github.com/teammateusername)


for their contributions to the project.

---

## Algorithm Explanation
TODO:
[] Set A , Kushal should add
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