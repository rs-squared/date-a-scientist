"""
Created by: rb0 
Template and Instructions from: Codecademy/Machine Learning Course
Date of Submition: 12/04/2018

"""
#%% Import libraries and objects used
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time
from sklearn.svm import SVC

#%% Close all plots
plt.close("all")

#%% Load Data and Explore columns 
df=pd.read_csv("profiles.csv")
df=df.dropna(axis=0)

print(df.pets.value_counts())
print(df.body_type.value_counts())
print(df.diet.value_counts())
print(df.drinks.value_counts())
print(df.drugs.value_counts())
print(df.smokes.value_counts())

#%% Add columns of data for drugs, drinks, body type, and diet
drink_mapping = {"not at all":0,"rarely":1,"socially":2, "often":3,"very often":4,"desperately":5}
drug_mapping={"never":0,"sometimes":1,"often":2}
smokes_mapping={"no":0,"sometimes":1,"when drinking":2,"yes":3,"trying to quit":4}

df['drink_code']=df.drinks.map(drink_mapping)
df['drug_code']=df.drugs.map(drug_mapping)
df['smokes_code']=df.smokes.map(smokes_mapping)
df['diet_code']=pd.np.where(df.diet.str.contains('anything'),0,pd.np.where(df.diet.str.contains('vegetarian'),2,pd.np.where(df.diet.str.contains('other'),1,3)))
df['diet_binary']=pd.np.where(df.diet.str.contains('anything'),0,1)
df=df.dropna(axis=0,how='any')

#%% Data Exploration and Visualization
plt.figure(1) #drug and diet frequency
ax1=plt.subplot(2,2,1)
ax1.hist(df.drug_code)
plt.xticks([0,1,2])
plt.ylim(0, 3500)
ax1.set_xticklabels(drug_mapping.keys(),rotation=45)
plt.title("Frequency of Drugs")

ax2=plt.subplot(2,2,2)
ax2.hist(df.drink_code)
plt.xticks([0,1,2,3,4,5])
ax2.set_xticklabels(["not at all","rarely","socially", "often","very often","desperately"],rotation=45)
plt.ylim(0, 3500)
plt.title("Frequency of Drinks")

ax3=plt.subplot(2,2,3)
ax3.hist(df.smokes_code)
plt.xticks([0,1,2,3,4,5])
ax3.set_xticklabels(["no","sometimes","when drinking","yes","trying to quit"],rotation=45)
plt.ylim(0, 3500)
plt.title("Frequency of Smokes")

ax4=plt.subplot(2,2,4)
ax4.hist(df.diet_code)
plt.xticks([0,1,2,3])
ax4.set_xticklabels(["anything","other","vegetarian","morally dictated"],rotation=45)
plt.ylim(0, 3500)
plt.title("Frequency of Diets")

plt.subplots_adjust(wspace=0.2, hspace=0.4)

plt.figure(2) #smokes and diet correlation
heatmap, xedges, yedges = np.histogram2d(df.diet_code,df.smokes_code, bins=[5,4])  
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]  
plt.clf()
plt.imshow(heatmap, extent=extent)  
plt.title("Smokes and Diet Code Correlation")
plt.xlabel("Diet Code")
plt.xticks([0,1,2,3])
plt.ylabel("Smokes Code")
plt.yticks([0,1,2,3,4])
plt.colorbar()

plt.figure(3) #age frequency by groups
heatmap, xedges, yedges = np.histogram2d(df.diet_code,df.drug_code, bins=[3,4])  
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]  
plt.clf()
plt.imshow(heatmap, extent=extent)  
plt.title("Drugs and Diet Code Correlation")
plt.xlabel("Diet Code")
plt.xticks([0,1,2,3])
plt.ylabel("Drugs Code")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(4) #age frequency by groups
heatmap, xedges, yedges = np.histogram2d(df.diet_code,df.drink_code, bins=[6,4])  
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]  
plt.clf()
plt.imshow(heatmap, extent=extent)  
plt.title("Drinks and Diet Code Correlation")
plt.xlabel("Diet Code")
plt.xticks([0,1,2,3])
plt.ylabel("Drinks Code")
plt.yticks([0,1,2,3,4,5])
plt.colorbar()

plt.show()

#%% Define functions 
#generates the scores for regression models
def reg_score_card(model,trainx,trainy,testx,testy):
#    print("Train score:")
    train_score=model.score(trainx, trainy)
#    print(train_score) #view performance scores on training set
#    print("Test score:")
    test_score=model.score(testx,testy)
#    print(test_score) #view performance scores on testing set
    y_predict = model.predict(testx) #test the model and save results for graph
    plt.scatter(testy, y_predict)
    plt.xlabel("Diet Code as Recorded")
    plt.ylabel("Diet Code as Predicted")
    plt.xlim([-.1,3.1])
    plt.ylim([-0.1,3.1])
    plt.show()
    return(train_score,test_score)

#generates the scores for clustering models
def class_score_card(model, testx, testy):
    predicted_labels=model.predict(testx)
#    print("Score: %s",model.score(testx,testy))
#    print("Accuracy: %s", accuracy_score(testy,predicted_labels))
#    print("Recall: %s", recall_score(testy,predicted_labels))
#    print("Presision: %s",precision_score(testy,predicted_labels))
#    print("F1 Score: %s",f1_score(testy,predicted_labels))
    return(accuracy_score(testy,predicted_labels),
           recall_score(testy,predicted_labels),
           precision_score(testy,predicted_labels),
           f1_score(testy,predicted_labels))

#generates the plots for the comparisons of gamma, c, and scores
def plot_scatters(x,y,col):
    plt.scatter(x,y,c=col,cmap='viridis',s=400,vmin=0,vmax=1)
    plt.xlabel('Gama Values')
    plt.ylabel('C Values')
    plt.colorbar()
    plt.show
    
#%% Regression modeling
ylin=df[['diet_code']]#section out diet which is the y values for both single and multi

#Set up and score a single Linear Regression Model
xlin=df[['drink_code']] #section out the x values for linear regression model
#break out training and testing sets
xlin_train,xlin_test,ylin_train,ylin_test = train_test_split(xlin,ylin,train_size=0.8,test_size=0.2,random_state=42)

start1=time.time()
lr=LinearRegression() #set up Linear Regression with a shorter variable name
lr.fit(xlin_train,ylin_train) #train the model
print("Single Linear Regression")
plt.figure(5)
plt.title("Single Linear Regression of Diet and Drinks")
reg_score_card(lr,xlin_train, ylin_train,xlin_test, ylin_test)
end1=time.time()
print("time = %s"%(end1-start1))

#set up and score a Multiple Variable Linear Regression Model
xmlin=df[['drug_code','drink_code','smokes_code','age']] #section out multiple x's
xmlin_train,xmlin_test,ymlin_train,ymlin_test = train_test_split(xmlin,ylin,train_size=0.8,test_size=0.2,random_state=42)
start2=time.time()
mlr=LinearRegression() #set up Linear Regression with a shorter variable name
mlr.fit(xmlin_train,ymlin_train) #train the model
print("Multiple Linear Regression")
plt.figure(6)
plt.title("Multiple Linear Regression of Diet Code compared to Drinks, Smokes, Drugs and Age Codes")
reg_score_card(mlr,xmlin_train, ymlin_train,xmlin_test, ymlin_test) 
end2=time.time()
print("time = %s"%(end2-start2))

n_list=np.arange(2,40,1)
timez=[]
trainz=[]
testz=[]

for n in n_list:
    start3=time.time()
    reg=KNeighborsRegressor(n_neighbors=n,weights='distance')
    reg.fit(xmlin_train,ylin_train)
#    print("K Nearest Neighbor Regression")
    plt.figure(7)
    plt.title("K Nearest Neighbor Regression of Diet Code compared to Drinks, Smokes, Drugs and Age Codes")
    train_score,test_score=reg_score_card(reg,xmlin_train, ymlin_train,xmlin_test, ymlin_test) 
    end3=time.time()
#    print("time = %s"%(end3-start3))
    trainz.append(train_score)
    testz.append(test_score)
    timez.append((end3-start3))

plt.figure(8)
plt.plot(n_list,timez,color='red',label="Time")
plt.plot(n_list,trainz,color='blue',label="Train Score")
plt.plot(n_list,testz,color='black',label="Test Score")
plt.title("Scores and Time vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Scores and Time")
plt.legend()
plt.show()


#%%Classification Techniques
yclus=df[['diet_binary']]
xclus=df[['drug_code','drink_code','smokes_code','age']]
xclus_train,xclus_test,yclus_train,yclus_test = train_test_split(xclus,yclus,train_size=0.8,test_size=0.2,random_state=42)

#Support Vector Machine
gammaz=[]
cz=[]
accuracyz=[]
f1z=[]
recallz=[]
precisionz=[]
timez2=[]

for ig in range(1,50,2):
    for ic in range(1,50,2):
        start4=time.time()
        svcclass=SVC(kernel='rbf',gamma=ig,C=ic)
        svcclass.fit(xclus_train,yclus_train)
        ac,rcz,pr,f1=class_score_card(svcclass,xclus_test,yclus_test)
        end4=time.time()
        gammaz.append(ig)
        cz.append(ic)
        accuracyz.append(ac)
        recallz.append(rcz)
        precisionz.append(pr)
        f1z.append(f1)
        timez2.append((end4-start4))

plt.figure(9)
plt.title('SVC Accuracy Compared to C and Gamma')
plot_scatters(gammaz,cz,accuracyz)

plt.figure(10)
plt.title('SVC Recall Compared to C and Gamma')
plot_scatters(gammaz,cz,recallz)

plt.figure(11)
plt.title('SVC Precision Compared to C and Gamma')
plot_scatters(gammaz,cz,precisionz)

plt.figure(12)
plt.title('SVC F1 Score Compared to C and Gamma')
plot_scatters(gammaz,cz,f1z)

plt.figure(13)
plt.title('SVC Time Compared to C and Gamma')
plot_scatters(gammaz,cz,timez2)

# Set up K-Nearest Neighbor
nz=[]
accuracyz2=[]
f1z2=[]
recallz2=[]
precisionz2=[]
timez3=[]

for nx in range(2,20,2):
    start5=time.time()
    knear=KNeighborsClassifier(n_neighbors=nx)
    knear.fit(xclus_train,yclus_train)
    ac,rc,pr,f1=class_score_card(knear,xclus_test,yclus_test)
    end5=time.time()
    nz.append(nx)
    accuracyz2.append(ac)
    recallz2.append(rc)
    precisionz2.append(pr)
    f1z2.append(f1)
    timez3.append((end5-start5))

fig,ax1=plt.subplots()
ax1.set_title('Scores and Time vs. Number of Neighbors')
ax2=ax1.twinx()
ax1.set_xlabel('Number of Neighbors')
ax1.set_ylabel('Scores')
ax2.set_ylabel('Time')
ax1.plot(nz,accuracyz2,color='red',label='Accuracy')
ax1.plot(nz,f1z2,color='blue',label='F1 Score')
ax1.plot(nz,recallz2,color='brown',label='Recall')
ax1.plot(nz,precisionz2,color='violet',label='Precision')
ax2.plot(nz,timez3,color='black',label='Time')
fig.legend(loc=1)

#%%### Print average times for powerpoint
print("Average Time for SVC = %s"%np.mean(timez2))
print("Average Time for KNearstClass = %s"%np.mean(timez3))

