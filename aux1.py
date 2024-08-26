from ucimlrepo import fetch_ucirepo

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
# print(mushroom.metadata)

print("ELDDDD \n\n")
# variable information 
print(mushroom.variables)