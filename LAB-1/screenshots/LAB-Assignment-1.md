# Python and Deep learning programming LAB-1

### Team members
Sindhusha Tiyyagura    CLASS ID: 37    
Pradeepika Kolluru        CLASS ID: 15   
Sravan Kumar Pagadala    CLASS ID: 31    

## Introduction
This report is about python and machine learning basic concepts

## Objective
The main objective of this lab work is to get handsome experience on some of the concepts in python like lists, tuples, sets, dictionaries, performing various string operations and OOPS concepts. This lab work also contains the application of some of the machine learning algorithms like Navie Bayes, SVM, KNN, Multiple regression and K-Means clustering.

## Problem-1 : Creating a dictionary with key as names and values as list of (subjects, marks) in sorted order
### Approach 
* The input is given in the program itself ( input --> list of tuples )   
* Initializing a dictionary with list elements as values.
* Storing the values of the list into a dictionary with key as name and (subject, marks) tuple as the value in the list.
* Each value in dict_values is a list of tuples. Sorting the list based on the name field in the tuple and storing them in another dict (sorted_dict)
* Displaying the dictionary after sorting based on name and dumping the value in the JSON format with 4 as indentation.
### Workflow
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-1%20Sorted%20dictionary/Task-1%20code.PNG)   
### Output
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-1%20Sorted%20dictionary/Task-1%20output.PNG)


## Problem-2 : Finding longest substring without repeating characters
### Approach
* Read the input string(str) from console
* lengthOflongestSubString() function is defined which prints the longest substrings without repeating characters with their lengths. str is passed as input parameter to the function.
* This function is created inside the class function Solution class.
* The longest Substrings can be done in single parse.
* The following code has the start index and the end index -> where the characters inside the start and end indexes don't have any repetition of characters.
* while extending the string in one direction -> if there is any repetition with the current character then start index is incremented after storing the string in the list of sub strings.
* If the character doesn't match any string the previous sub string then the end index is incremented.
* Later finding the max length of the sub string inside the list elements.
* printing all the longest sub strings. 
### Workflow
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-2%20longest%20substring/Task-2%20code.PNG)
### Output
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-2%20longest%20substring/Task-2%20output.PNG)


## Problem - 3 : Airline Booking Reservation System
### Approach
We have created five classes Flight, passenger, Employee, Fare, Baggage, Ticket. Each class members are intialized by using init constructor.
**Flight class** : Flight class contains the flight details like Airlines and flight number.    
**Employee Class** : In the employee class the data members are initialized during object creation. emp_display() displays the details of employee like name, age, emp_id, gender.     
**Passenger class** : In Passenger class the data members are initialized by reading the input from the console.    
**Baggage Class** : Cabin_bag and bag_fare are class the class members, Checked bag is the data member which is initialized while object creation. If the number of checked bags are greater than 2 then bag_fare is increased by 100.      
**Fare Class** : Fare class inherits the Baggage class. It generates the fare by including all costs like bag_fare, transaction type.    
**Ticket Class** : Ticket class shows multiple inheritance. It inherits both Passenger and Fare class. Fare class inherits the Baggage class.Depending upon the class type it gives the overall fare for the itinerary.    
### Workflow
![](https://github.com/pradeepika1997/Python-Deep-learning-programming/raw/master/LAB-1/Screenshots/Screenshot%20(189).png)
### Output
![](https://github.com/pradeepika1997/Python-Deep-learning-programming/raw/master/LAB-1/Screenshots/Screenshot%20(190).png)


## Problem - 4 : Multiple Regression
### Approach
* Boston dataset is chosen from pandas
* Data is checked for the null values. If there are any null values then they will be replaced by mean. In this data set there are no null values.
* Finding the correlation between the target class MEDV and features. Most correlated features are used and remaining all features are dropped.
* Splitting the data into training and test data using train_test_split
* Creating the regression model and training it by providing the train data
* Predicting the target class. In multiple regression, model evaluation is performed by using Mean square error and r2_score
* Calculating the Mean Square Error and R2 score
* A multiple regression model is evaluated as good if its Mean Square Error is low and R2_score is high. What we observed from this program is before performing EDA Mean Square Error is high and R2 score is low. After performing EDA, Mean Square Error is low and R2_score is high.
### Workflow
![](https://github.com/pradeepika1997/Python-Deep-learning-programming/raw/master/LAB-1/Screenshots/Screenshot%20(191).png)
### Output
![](https://github.com/pradeepika1997/Python-Deep-learning-programming/raw/master/LAB-1/Screenshots/Screenshot%20(192).png)


## Problem - 5 : Comparing Different Classifications accuracy
### Approach
* Importing all required libraries
* Loading telecom customer details dataset using pandas library
* Replacing empty values in the Total Charges column with the NaN value and then dropping the fields with NaN value.
* dropping Customer ID column which is not necessary for classification as it a random number.
* Finding the correlation of each feature with other all features and plotting the heat map graph which shows the correlation with different shades of colors
* dropping all the columns which are having very less correlation
* converting all object datatypes ( categorical) to the numerical data type.
* Based on RFE feature selection ranking splitting the dataset into train and test dataset.
* Applying 3 different models and training the models with X_train and Y-train datasets.
* predicting the ouput y_predict for all 3 models.
* Using the y_predict and y_test data -> calculating the accuracy score provided by the metric sklearn library.
* After evaluating the scores -> kNN has 0.7 accuracy
* SVM linear has 1 accuracy score.
* where as naive bayes classification has 1 accuracy score.
### Workflow
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-5%20Classification/code.PNG)
### Output
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-5%20Classification/Task-5%20output%20-1.png)
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-5%20Classification/task-5%20output.PNG)


## Problem - 5 : Comparing Different Classifications accuracy
### Approach
* Importing all required libraries
* Loading College details dataset using pandas library
* Splitting the features/columns based on indexes
* printing the null values in the dataset for any column
* replacing the null values with the mean value
* Splitting the dataset and Applying the K means clustering on the dataset.
* predicting the test data using the build/trained model.
* Using the predicted score finding the Silhoutte score where it was 0.59 score. 
* elbow method shows that k=2 value is the number of cluster from the elbow graph.
### Workflow
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-6%20Clustering/code.PNG)
### Output
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-6%20Clustering/output-1.png)
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-6%20Clustering/output-2.png)
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-6%20Clustering/output-3.png)
![](https://github.com/sindhusha-t/Python-Programming/raw/master/LAB-1/screenshots/Task-6%20Clustering/Task-6%20output.PNG)