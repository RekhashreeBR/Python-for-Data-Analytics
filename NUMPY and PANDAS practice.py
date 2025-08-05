# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:24:31 2024
NUMPY  AND PANDAS PRACTICAL QUESTIONS
@author: hp
"""
import numpy as np

# =============================================================================
# #1.Create a 3x3 NumPy array filled with
#    random integers and print its shape and data type.
# =============================================================================
a=np.random.randint(0, 100,(3,3))
print(a)
print('\n')


# =============================================================================
# #2. Convert a Python list of lists into a
#      NumPy ndarray and print the resulting array.
# =============================================================================
b=[[1,2],[3,6]]
print('\n')
print(b)
b1=np.asarray(b)
print(b1)

# =============================================================================
# 3. Create a 5x5 NumPy array of zeros and 
#    change the element at position (2, 2) to 5.
# =============================================================================
c=np.zeros((5,5))
print(c)
c[2,2]=5
print(c)
print('\n')

# =============================================================================
# Create a 4x4 identity matrix using NumPy.
# =============================================================================
d=np.identity(4)
print(d)
print('\n')

# =============================================================================
# 5. Create a NumPy array of shape (3, 3, 3)
#    filled with random values and print the array.
# =============================================================================
e=np.random.randint(1,100,(3,3,3))
print(e)
print('\n')

# =============================================================================
# 6. Create a 10x10 NumPy array of ones 
#    and set the boundary values to zero (except the boundary).
# =============================================================================
f=np.ones((10,10))
print(f)
print('\n')
f[0,0]=0
f[0,-1]=0
f[9,0]=0 
f[9,-1]=0
print(f)
print('\n')

# =============================================================================
#7. Generate a NumPy array of 50 linearly spaced numbers between 0 and 100. 
# =============================================================================
g=np.linspace(0, 100,50)
print(g)
print('\n')

# =============================================================================
# 8. Convert a NumPy array of integers into a float array.
# =============================================================================
h=np.array([1,2,346,76,45])
print("int:",h)
h1=h.astype(float)
h1=np.array(h,dtype=float)
print("float:",h1)
print()


# =============================================================================
# 9. Create a NumPy array of size 20 and reshape it to (5, 4).
# =============================================================================
i=np.arange(20)
i1=np.reshape(i,(5,4))
print(i1)
print('\n')

# =============================================================================
# 10. Create a 2D NumPy array of shape (5, 5) 
#     and calculate the sum of all elements.
# =============================================================================
j=np.arange(25)
j1=np.reshape(j,(5,5))
jsum=j1.sum()
print(jsum)

print('\n')
# =============================================================================
# 11. Create a 6x6 NumPy array and extract the subarray 
#     of the first 3 rows and 3 columns.
# =============================================================================
k=np.arange(36)
k1=np.reshape(k,(6,6))
print(k1)
subk1row=k1[:3,:-1] 
subk1col=k1[:5,:3]
print(subk1row)
print(subk1col)
print('\n') 


# =============================================================================
#12. Create a 1D NumPy array of size 10 and
# extract every other element starting from index 0. 
# =============================================================================
l=np.arange(20,40,2 )
print(len(l))
print(l[0:])

print('\n')

# =============================================================================
# 13. Reverse the elements of a 1D NumPy array using slicing.
# =============================================================================
m=np.arange(20,40,2 )
print(m[::-1])

print('\n')

# =============================================================================
# 14. Create a 5x5 array and replace all the odd elements with -1.
# =============================================================================
n=np.arange(25)
n1=np.reshape(n,(5,5))
print(n1)
n1[n1%2!=0]=-1
print(n1)

print('\n')
 
# =============================================================================
# 15. Select the elements in an array that are
#     greater than a given threshold value.
# =============================================================================

a=28
p=np.random.randint(10,40,(2,3))
print(p)
print('\n')
print(p[p>a])
print('\n')

# =============================================================================
# 16. Use fancy indexing to select specific rows 
#     and columns from a 2D NumPy array.
# =============================================================================
q=np.arange(4)
q1=np.reshape(q,(2,2))

print(q1)

print('\n')
# =============================================================================
# 17. Create a 3D array and extract a subarray using multidimensional slicing.
# =============================================================================
r=np.arange(27)
r=np.reshape(r,(3,3,3))
print(r)
print('\n')
print(r[0:3][...,1][2,...])

# =============================================================================
# 18. Modify the last row of a 4x4 NumPy array to contain all ones.
# =============================================================================
s=np.arange(16)
s=np.reshape(s,(4,4))
print(s)
s[3]=1
print(s)

print('\n')


# =============================================================================
# 19. Extract the diagonal elements from a 2D square NumPy array.
# =============================================================================
t=np.arange(4)
t= np.reshape(t,(2,2))
print(t)
t1=np.diag(t)
print(t1)

print('\n')

# =============================================================================
# 20. Using Boolean indexing, extract all elements 
#     of an array that are even numbers.
# =============================================================================
u=np.arange(0,9)
u1=u%2==0
print(u1)
print(u[u1])

print('\n')

# =============================================================================
# 21. Create a 2D NumPy array and transpose it.
# =============================================================================
v=np.array([[32,4],
            [32,45]])
print(np.shape(v))
print(np.transpose(v))

print('\n')

# =============================================================================
# 22. Create a matrix and compare its transpose with the original matrix.
# =============================================================================
w=np.matrix([[32,45,67],
             [452,54,67],
             [314,5,57]])
print(w)
wt=np.transpose(w)
print(wt)

print('\n')
# =============================================================================
# 23. Create a 3x3 matrix and swap the rows with columns using transposition.
# =============================================================================

x=np.matrix([[32,45,67],
             [452,54,67],
             [314,5,57]])
print(x)
print(np.swapaxes(x,1,0))

print('\n')

# =============================================================================
# 24. Transpose a NumPy array of shape (4, 2) and reshape it to (2, 4).
# =============================================================================
y=np.array([[32,345],
            [31,42],
            [42,42],
            [21,13]])
print(np.shape(y))
y1=np.transpose(y)
print(y1)
print(np.reshape(y1,(2,4)))

print('\n')

# =============================================================================
# 25. Create a 5x5 matrix, swap the first and second rows, and print the result.
# =============================================================================
z=np.matrix([[1,2,3,4,5],
             [6,7,8,9,0],
             [0,9,8,7,6],
             [5,4,3,2,1],
             [4,4,2,5,2]])
z[[0,2]]=z[[1,1]]
print(z)

print('\n')
# =============================================================================
#26. Write a function that accepts a NumPy array 
#    and returns its transpose using np.transpose().
# =============================================================================
def Transpose(x):
    return np.transpose(x)
a=np.arange(12)
b=np.reshape(a,(4,3))
c=Transpose(b)
print(c)
print('\n')
# =============================================================================
# 27. Create a NumPy array and save it to a .npy file using np.save().
# =============================================================================
d=np.array([1,2,3,4,5,6,7,8,9,0])
np.save("array.npy",d)

# =============================================================================
# 28. Create a random array and save it as a .csv file using np.savetxt().
# =============================================================================
np.savetxt("array.csv", d,delimiter="<")


# =============================================================================
# 29. Load a NumPy array from a .npy file using np.load().
# =============================================================================
loading_array=np.load("array.npy")
print(loading_array)



# =============================================================================
# 30. Save multiple NumPy arrays into a single .npz file and then load them.
# =============================================================================
e=np.array(["wswr","rfgs","rgfs","sdrg"])
f=np.array([1,2,3,4,5,6,7,8,9,0])
np.savez("arrays.npz", arr1=e,arr2=f)
#loading
loading_arrays=np.load("arrays.npz")
print(loading_arrays["arr1"])
print(loading_arrays["arr2"])
print('\n')




# =============================================================================
# 31. Create a 3x3 matrix and save it to a text file,
#     then reload it and print the contents.
# =============================================================================
g=np.matrix([[34,57,87],
             [87,8754,23],
             [32,78,35]])
np.savetxt("matrix.csv", g)
loading_matrix=np.loadtxt("matrix.csv")
print(loading_matrix)
print('\n')



# =============================================================================
# 32. Load a .csv file into a NumPy array and print the first 5 rows.
# =============================================================================
#h=np.loadtxt('titanic_dataset.csv',dtype=int)

# =============================================================================
# 33. Create two NumPy arrays, save them in binary format, 
#      and then reload and print them.
# =============================================================================
e=np.array(["wswr","rfgs","rgfs","sdrg"])
f=np.array([1,2,3,4,5,6,7,8,9,0])
np.save("array1", e)   #not saving by npy extention to let it in binary
i=np.load("array1.npy")
np.save('array2', f)
j=np.load("array2.npy")
print(i)
print(j)
print('\n')




# =============================================================================
# 34. Save a NumPy array in compressed format using 
#      np.savez_compressed() and reload it.
# =============================================================================
k=np.array([1,2,3,4,5,6,7,8,9,0])
np.savez_compressed("array_compressed", arr=k)
loading_compressed_arr=np.load("array_compressed.npz")
print(loading_compressed_arr["arr"])
print('\n')



# =============================================================================
# 35. Generate a large NumPy array save it to a file and measure the file size. 
# =============================================================================
l=np.arange(0,10000)
j=np.savetxt("large_arr", l)
import os
arr_size=os.path.getsize("large_arr")
print(arr_size)
print('\n')




# =============================================================================
# 36. Write a function that saves an array to a file and then loads it back.
# =============================================================================
def save_arr(name,arr):
    np.savetxt(name, arr)
    n=np.loadtxt("arr")
    return n
n=np.array([[324,57,46],
            [254,356,375]])
fnk=save_arr("arr",n )
print(fnk)
print('\n')




# =============================================================================
# 37. Create a NumPy array of random values and apply the np.sqrt()
#     function to compute the square root of each element.
# =============================================================================
m=np.random.randint(0,100,50)
sqrt_m=np.sqrt(m)
print(sqrt_m)
print('\n')



 
# =============================================================================
# 38. Generate a NumPy array of 100 random values and 
#    apply np.sin() to compute the sine of each element. 
# =============================================================================
o=np.random.randint(60,200,100)
sin_0=np.sin(o)
print(sin_0)

print('\n')



# =============================================================================
#39. Given a NumPy array of floating-point numbers,
# apply the np.ceil() function to round each value up. 
# =============================================================================
p=np.array([[54.55,45.4679,76.3421],
            [7865.6754,7654.54,765.8976543]])
print(np.ceil(p))

print('\n')



# =============================================================================
# 40. Create a NumPy array and apply np.exp() to compute the exponential of 
#     each element.
# =============================================================================
k=np.array([1,2,3,4,5,6,7,8,9,0])
print(np.exp(k))
print('\n')



# =============================================================================
# 41. Create two arrays and use np.maximum() to compute 
#     the element-wise maximum of the two arrays.
# =============================================================================
q=np.array([23,3554,87])
r=np.array([876,2,67])
print(np.maximum(q,r))
print('\n')



# =============================================================================
# 42. Create a 2D array of random integers and apply np.abs()
# to get the absolute values of all elements.
# =============================================================================
s=np.random.randint(38,40,6)
t=np.reshape(s, (3,2))
print(t)
print('\n')
print(np.abs(t))
print('\n')


# =============================================================================
# 43.Apply np.log() to a NumPy array of positive random values 
#     and print the results.
# =============================================================================
k=np.array([1,2,3,4,5,6,7,8,9])
print(np.log(k))



# =============================================================================
# 44. Create a NumPy array of random floats and apply np.round() 
#    to round each value to 2 decimal places.
# =============================================================================
u=np.random.randint(1,5,20)
print(u/21)
print('\n')
print(np.round(u,decimals=2))
print('\n')


# =============================================================================
# 45. Use np.add() to add two NumPy arrays element-wise.
# =============================================================================
k=np.array([1,2,3,4,5,6,7,8,9])
l=np.array([50,60,70,80,90,40,30,10,20])
print(np.add(k,l))
print('\n')



# =============================================================================
# 46.Apply np.power() to raise each element in an array to a given power.
# 
# =============================================================================
k=np.array([1,2,3,4,5,6,7,8,9])
print(k)
print(np.power(k,3))
print('\n')


# =============================================================================
#47.Create a NumPy array of random integers and compute mean using np.mean().

# =============================================================================
l=np.array([50,60,70,80,90,40,30,10,20])
print(np.mean(l))

print('\n')

# =============================================================================
#48. Generate a 2D array and compute the sum of all its elements using np.sum()
# =============================================================================
r=np.array([[1,2,98,4],
            [5,67,97,8]])
sum_r=np.sum(r)
print(sum_r)




print('\n')
# =============================================================================
# 50. Create a NumPy array and compute the median of its values
#        using np.median()
# =============================================================================
r=np.array([[1,2,98,4],
            [5,67,97,8]])
print(np.median(r))
print('\n')



# =============================================================================
#51.Create a 3x3 matrix and compute the trace (sum of diagonal elements) using
# np.trace().
# =============================================================================
v=np.array([[45,67,89],
            [23,45,868],
            [28,73,65]])
np.asmatrix(v)
print(np.trace(v))
print('\n')



# =============================================================================
# 52. Create a 2D array of random values and compute the row-wise and
#     column-wise sums.
# =============================================================================
v=np.array([[4,6,9],
            [3,5,8],
            [2,3,5]])
print(v)
import pandas as pd
jk=pd.DataFrame(v)
print(jk)
print("row sum=",np.sum(v,axis=1))
print("column sum=",np.sum(v,axis=0)) 
print('\n')
print("row sum=",np.sum(jk,axis=1))
print("column sum=",np.sum(jk,axis=0))


# =============================================================================
# 53. Given a 2D array, calculate the minimum and
#     maximum values along both axes using
# =============================================================================
v=np.array([[45,67,89],
            [23,45,868],
            [28,73,65]])
print(np.min(v,axis=1))
print(np.max(v,axis=0))
print('\n')


# =============================================================================
#  54.Compute the variance of a NumPy array using np.var()
# =============================================================================
v=np.array([[45,67,89],
            [23,45,868],
            [28,73,65]])
print(np.var(v))






# =============================================================================
# pandas
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
#1. Create a pandas Series from a list of numbers and display it. 
# =============================================================================
a=pd.Series([23,56,78,87,54,34])
print(a)
print('\n')



# =============================================================================
# 2. Create a pandas Series from a Python dictionary, where keys represent
#    index values and values represent data.
# =============================================================================
b=pd.Series({"a":20,"b":30,"c":40})
print(b)
print('\n')



# =============================================================================
# 3. Given a pandas Series of exam scores, calculate the mean, median, and mode 
# =============================================================================
c=pd.Series({"python":15,"math":15,"eco":17,"eng":16,"lan":18})
print(c)
mean=c.mean()
print("mean=",mean)
median=c.median()
print("median=",median)
mode=c.mode()
print("mode=",mode)
print('\n')



# =============================================================================
# 4. Create a DataFrame from a dictionary of lists and
#    display the first five rows. 
# =============================================================================
d=pd.DataFrame({"name":["rekha","sanjana","yogana","thag",
                        "paddu","anitha","bby"],
             "age":[18,17,19,21,40,19,20]})
print('\n')
print(d.head())
print('\n')



# =============================================================================
# 5. Create a DataFrame with custom row and column labels using numpy arrays.
# =============================================================================
e=np.array(["a","b","c","d"])
f=np.array(["1st","2nd","3rd"])
fe=np.random.randint(12,45,(4,3))
ef=pd.DataFrame(fe,index=e,columns=f)
print(ef)
print('\n')

# =============================================================================
# 6. Create a DataFrame where one of the columns is a pandas Series, and 
#     other columns are lists. 
# =============================================================================
g=pd.Series([23,56,78])
h=["awsd","rdfgh","asdfg"]
gh=pd.DataFrame({"series":g,"list":h})
print(gh)
print('\n')


# =============================================================================
# 7. Convert a pandas Series to a DataFrame with a custom column name. 
# =============================================================================
g=pd.Series([23,56,78])
print("series:\n",g)
gd=pd.DataFrame(g,columns=["marks"])
print("dataframe:\n",gd)
print('\n')



# =============================================================================
# 8.Compare two Series element-wise and print the values that are equal in both
# =============================================================================
g=pd.Series([23,56,78,32,45])
print("series 1:",g)
j=pd.Series([23,54,78,54,57])
print("series2:",j)
print('\n')
print("value that are equal in both:\n",g[g==j])
print('\n')



# =============================================================================
# 9. Create a DataFrame with NaN values and fill them with the mean of each
#    column. 
# =============================================================================
h=pd.DataFrame({"math":[10,np.nan,37,68,25,6],"eco":[34,78,34,np.NaN,53,43]})
print("data with na:")
print(h)
print('\n')
print("filling the data with mean values:")
print('\n')
hp=h.apply(lambda n:n.fillna(n.mean()))
print(hp)
print('\n')



# =============================================================================
# 10. Create a DataFrame from a nested list, assign column names, and 
#     add a new column to the DataFrame. 
# =============================================================================
K=[[23,34,56],[45,23,45],[56,23,88]]
print("nested list: \n",K)
print('\n')
dk=pd.DataFrame(K,columns=["a","b","c"])
print("dataframe of nested list:\n",dk)
dk["d"]=[43,56,87]
print("adding new column\n",dk)
print('\n')


# =============================================================================
# 11. Create a DataFrame with date ranges as the index and 
#     random numbers as values. 
# =============================================================================
date=pd.date_range('05/10/2024',periods=5)
d=np.random.randint(0,8,5)
print(d)
ddate=pd.DataFrame(d,index=date)
print("\ndata frame of random valus with date range as index:",ddate)
print('\n')



# =============================================================================
# 12. Generate a DataFrame from a dictionary where the keys are column names
#     and values are lists of numbers. 
# =============================================================================
p={"abc":[1,2,3,4,5,6,7,8,9],"def":[1,3,5,7,9,2,4,6,8]}
print("the dictionary is\n",p)
print('\n')
ph=pd.DataFrame(p)
print("dictionary with keys as columnb names (by default) : \n",ph)
print('\n')

# =============================================================================
# 13. Create a DataFrame from two pandas Series as two columns. 
# =============================================================================
v=pd.Series([45,45,456,456,56,45])
w=pd.Series([23,2,23,35,789,9])
vw=pd.DataFrame({"s1":v,"s2":w})
print("printing 2 series in column :\n",vw)
print('\n')



# =============================================================================
# 14. Create a DataFrame with multiple indexes (MultiIndex) and 
#     retrieve data from it.
# =============================================================================

index1=[['male','male','female','female'],
        ['gagan','rohan','rekha','swathi']]
MultiIndexi=pd.MultiIndex.from_arrays(index1,names=('student','name'))
mn=pd.DataFrame({0:['pass','pass''fail','fail']},index=MultiIndexi)
mn


# =============================================================================
# 15. Access a specific column from a DataFrame by name and return it as Series
# =============================================================================
print("the data frame is :\n",vw)
print("accssing the 1st col:\n",vw["s1"])
print("in series:\n",pd.Series(vw["s1"]))
print('\n')




# =============================================================================
# 16. Select multiple columns from a DataFrame and return them as new DataFrame. 
# =============================================================================
kl=pd.DataFrame({'h':[3,5,7,8],'j':[3,6,9,3],'g':[5,3,9,2],'y':[7,3,9,5]})
print("the data frame is :\n",kl)
newdf=pd.DataFrame(kl[["j","g"]])
print("the new data frame:\n",newdf)

print('\n')
# =============================================================================
# 17. Access a column from a DataFrame using dot notation (i.e.,df.column_name). 
# =============================================================================
print("printing the 'y' col using .notation :\n",kl.y)
print('\n')



# =============================================================================
# 18. Add a new column to a DataFrame using existing columns 
#     e.g., sum two columns). 
# =============================================================================
kl=pd.DataFrame({'h':[3,5,7,8],'j':[3,6,9,3],'g':[5,3,9,2],'y':[7,3,9,5]})
kl["gy"]=kl["g"]+kl["y"]
print("added a new col of sum of 'g' and 'y' :\n",kl)
print('\n')


# =============================================================================
# 19. Rename a column in a DataFrame and display the updated DataFrame. 
# =============================================================================
print("the data frame is :\n",kl)
new_kl=kl.rename(columns={"gy":"sum_gy"})
print("renamed df:\n",new_kl)
print('\n')


# =============================================================================
# 20. Check if a particular column exists in a DataFrame and
  #    print a message accordingly. 
# =============================================================================
print("the data frame is :\n",kl)
col="j"
if col in kl:
    print("the column",col," exist")
else:
    print("the column ",col,"doesnt exist")
print('\n')


# =============================================================================
# 21. Select only the numerical columns from a DataFrame and
#      display their data types. 
# =============================================================================
bv=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20]})
print("the df is:\n",bv)
bv_int=bv.select_dtypes(include=int)
print("df of only col with int vals :\n",bv_int)
print('\n')



# =============================================================================
# 22. Use the .iloc[] method to select the first column from a DataFrame.
# =============================================================================
mn=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20]})
print(mn)
print("selecting the 1st col\n")
print(mn.iloc[0:,0])
print('\n')
# =============================================================================
# 23. Access the first row of a DataFrame using the .iloc[] method. 
# =============================================================================
print("first row of df is :\n",mn.iloc[0,0:])

print('\n')
# =============================================================================
# 24. Use .loc[] to access rows of a DataFrame based on index labels. 
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20]},index=["std1","std2","std3","std4"])
print(jk)
print('\n')
print("1st row of ddf is:\n",jk.loc["std1"])
print('\n')


# =============================================================================
# 25. Filter rows based on a condition
#      (e.g., values in a specific column greater than 50). 
# =============================================================================
kl=pd.DataFrame({'h':[3,5,7,8],'j':[3,6,9,3],'g':[5,3,9,2],'y':[7,3,9,5]})
print(kl)
print(kl[kl['j']<5])
print('\n')
# =============================================================================
# 26. Select the top 5 rows where a column contains a specific value
#     (e.g., 'Male' in a gender column). 
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]})
print(jk)
print('\n')
pk=jk[jk['gender']=="male"]
print("top 5 rows with gender 'male':\n",pk.head())

print('\n')
# =============================================================================
# 27. Retrieve rows where multiple conditions are true 
#         (e.g., column A > 50 and column B < 20). 
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]})
print(jk)
print('\n')
lo=jk[(jk['gender']=="male") &( jk["age"]>50)]
print("data of male above 50 yrs of age:'n",lo)
print('\n')

# =============================================================================
# 28. Select rows using a range of index labels 
#      (e.g., rows from index 'A' to 'D'). 
# =============================================================================
alp=pd.DataFrame([1,2,    3,4,5,6,7,8,9,0], 
                 index=["a","b","c","d","e","f","g","h","i","j"])
print("the data is: \n",alp)
print("ranging using alphabets:\n",alp['b':'h'])

# =============================================================================
# 29. Access rows of a DataFrame using a boolean mask and display the filtered 
# DataFrame.
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]})
print(jk)
boolean=jk['gender']=='female'
print(jk[boolean])
print('\n')
# =============================================================================
# 30. Create a pandas Index object from a list of values and 
#     use it to index a Series. 
# =============================================================================
index=["a","b","c","d"]
u=pd.Series(["apple","ball","cat","dog"],index=index)
i=u.reset_index()
print(i)
print("the series with the listr of index :\n",u)
print('\n')


# =============================================================================
# 31. Retrieve the index values of a DataFrame and print them. 
# =============================================================================
print("the index values of u is :\n",u.index)
print('\n')

# =============================================================================
# 32. Set a column as the index of a DataFrame and
#     display the updated DataFrame. 
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]})
print(jk)
print('\n')
print("set name as index :\n",jk.set_index("name"))
print('\n')

# =============================================================================
# 33. Reset the index of a DataFrame and
#     convert the index into a regular column. 
# =============================================================================
jk=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]})
print(jk)
print('\n')
print("set name as index :\n",jk.set_index("name"))
print('\n')
print("reseted index :\n",jk.reset_index())
print('\n')

# =============================================================================
# 34. Create a pandas RangeIndex and explain its difference
#       from a regular Index. 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya",
                         "jgft","jyghtd","iuytr","iluyt","jkhg"],
                 "age":[17,18,19,20,6,84,98,56,3],
                 "gender":["male","female","female",
                           "male","male","male","male","male","male"]},
                index=pd.RangeIndex(4,13))
print("the data frame given with the range index :\n",iu)
print('\n')
"""range index automatically give the index to the data with in the given
 range along with skipping th vale if mentiond .it snt required to mention
 each index numbers manually"""

# =============================================================================
# 35. Rename the index of a DataFrame and show the updated DataFrame. 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},
                index=[1,2,3,4])
print("dat frame with index :\n",iu)
print('\n')
oi=iu.rename(index={1:"a",2:"b",3:"c",4:"d"})
print(oi)
print('\n')
# =============================================================================
# 36. Check if a given label exists in the index of a DataFrame
# =============================================================================
if "c" in oi.index:
    print('c',"is present")
else:
    print('c is not present')

print('\n')

# =============================================================================
# 37. Reindex a Series using a list of new labels and
#      fill any missing values with 0. 
# =============================================================================
f= pd.Series(['a', 'b', 'c', 'd'])
d= [0, 1, 2, 3, 4, 5]
df= f.reindex(d, fill_value=0)
print(df)
print('\n')

# =============================================================================
# 38. Reindex a DataFrame with a new list of row labels and 
#     set a different order of columns. 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},
                index=[1,2,3,4])
print("original data frame: \n",iu)
b=[5,6,7,8]
h=iu.reindex(b)
print(h)


# =============================================================================
# 39. Reindex a DataFrame using a method that forward fills missing values 
#(e.g., method='ffill'). 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},index=[1,2,3,4])
print("original data frame: \n",iu)
b=[1,2,3,4,5,6,7,8]
h=iu.reindex(b).ffill()
print(h)


# =============================================================================
# 40. Reindex a DataFrame to match the index of another DataFrame. 
# =============================================================================
ag=pd.DataFrame([234,56,67,89])
rt=pd.DataFrame([898,76,545,54,88])
print(ag)
print(rt)
agrt=pd.concat([ag,rt],ignore_index=True)
print(agrt)
print('\n')


# =============================================================================
# 41.Align two DataFrames on their indexes and fill any missing values with NaN 
# =============================================================================
st=pd.DataFrame([[2,3],[5,7]])
op=pd.DataFrame([[34,54,65],[43,57,87]])
stop=pd.concat([st,op]).fillna(0)
print("filled with missing value\n: ",stop)
print("\n")



# =============================================================================
# 42. Reindex a DataFrame by reversing the index order and show the result. 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},index=[1,2,3,4])
print("original data frame: \n",iu)
dw=iu[::-1]
print('\n')
print("reversed data frame :\n",dw)


# =============================================================================
# 43. Reindex a Series and assign default values for missing entries using 
#     the .fillna() method.
# =============================================================================
hd=pd.Series([34,65,76,78,89,43])
print(hd)
hg=hd.reindex([1,2,3,4,5,6,7,8,9]).fillna(0)
print(hg)
# =============================================================================
# 44. Drop a specific row from a DataFrame by label and display the result. 
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},index=[1,2,3,4])
print("original data frame: \n",iu)
ik=iu.drop(2)
print(ik)

# =============================================================================
# 45. Drop multiple columns from a DataFrame by name using the .drop() method  
# display the result.
# =============================================================================
iu=pd.DataFrame({'name':["rekhashree","ramya","thanu","priya"],
                 "age":[17,18,19,20],
                 "gender":["male","female","female","male"]},index=[1,2,3,4])
print("original data frame: \n",iu)
ik=iu.drop(["name","gender"],axis=1)
print(ik)


# =============================================================================
# 
# =============================================================================
ipl={'team':['riders','riders','devils','devils',"kings","kings","kings","kings",'royals'
             ,'royals','royals','royals'],
     'rank':[1,2,2,3,3,4,1,1,2,4,1,2],
     'year':[2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
     'points':[876,788,836,673,746,667,876,788,812,891,981,888]
     }

df=pd.DataFrame(ipl)
df











