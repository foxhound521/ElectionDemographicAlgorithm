import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestRegressor
 
 
##################
# Read datasets in 
df = pd.read_csv("County_Facts.csv")
df1 = pd.read_csv("2016_votebycounty.csv")
df2 = pd.read_csv("2012prezdata.csv")
 
df = df[["area_name","state_abbreviation","POP010210","RHI825214","EDU685213",
         "LFE305213","INC910213","PVY020213","HSG445213","PST120214", "AGE135214", 
         "AGE295214", "AGE775214", "SEX255214","RHI225214", "RHI325214", "RHI425214", 
         "RHI525214", "RHI625214","RHI725214","POP715213","POP645213","POP815213",
         "EDU635213","VET605213","HSG096213","HSG495213","HSD310213","INC110213",
         "BZA010213","BZA110213","BZA115213","NES010213","BPS030214","POP060210"]]
df1 = df1[["Vote by County","Trump","Clinton","State"]]
df.columns = ['county', 'state', 'ppl','white','edu','commute','income','poverty','homeownerrate',
              'pop_chg','age5','ppl_under18','age65_over','female','blk','ind','asian','pacislander',
              'multirace','hisp','steadyhome','foreignborn','esl','hsgradrate','vet','apt','home_value',
              'ppl_per_house','household_income', 'biz','biz_emp','emp_chg','self_emp','building_permit',
              'pop_per_sqmile']
df1.columns = ['county','trump','clinton','state']
 
df2 = df2[["State","Name","Obama","Romney"]]
df2.columns = ['state','county','Obama','Romney']
 
#Normalize total numbers to rates
df['vet'] = df.vet/df.ppl
df['biz'] = df.biz/df.ppl
df['biz_emp'] = df.biz_emp/df.ppl
df['self_emp'] = df.self_emp/df.ppl
df['building_permit'] = df.building_permit/df.ppl
 
#Merge datasets
df = pd.merge(df, df1, on=['county', 'state']) #Join datasets on county and state
 
#Create vote totals
df['trumppct'] = df.trump/(df.trump+df.clinton) #Trump vote pct
df['clintonpct'] = (1-df.trumppct) #CLinton vote pct 
df.loc[df['trumppct'] >= .5, 'TrumpWin'] = 1 #Column for logistic regression - 1 = Trump Win
df.loc[df['trumppct'] < .5, 'TrumpWin'] = 0 #Column for logistic regression - 0 = Clinton Win
 
#Data cleanup
df = df[(df.trump > 0) & (df.clinton > 0)] #Remove any counties with no votes
df = df[df.state == df.state]
df = df[df.county == df.county]
 
df.dropna(how='any') #Drop NAs
 
predictor_columns = ['white','edu','commute','income','poverty','homeownerrate','pop_chg','age5','ppl_under18',
                     'age65_over','female','blk','ind','asian','pacislander','multirace','hisp','steadyhome',
                     'foreignborn','esl','hsgradrate','vet','apt','home_value','ppl_per_house','household_income', 
                     'biz','biz_emp','emp_chg','self_emp','building_permit','pop_per_sqmile']
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3) 
rf.fit(df[predictor_columns], df["TrumpWin"])
rfecv = RFECV(estimator=rf, step=1, cv=2, scoring='roc_auc', verbose=2)
x=pd.DataFrame(df, columns=['white','edu','commute','income','poverty','homeownerrate','pop_chg','age5',
                            'ppl_under18','age65_over','female','blk','ind','asian','pacislander','multirace',
                            'hisp','steadyhome','foreignborn','esl','hsgradrate','vet','apt','home_value',
                            'ppl_per_house','household_income', 'biz','biz_emp','emp_chg','self_emp',
                            'building_permit','pop_per_sqmile']) #algorith x var
 
y=(pd.Series(df.TrumpWin).astype(float))
selector=rfecv.fit(x, y)
print("\n" + "20/20 ANALYTICS Algorithm is processing. Please wait a few minutes. Thank you for your patience...")
new_predictor_columns = []
a = sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), predictor_columns))
for z in a:
    if(z[0])==1:
        new_predictor_columns.append(z[1]) #these are my new predictor variables
             
trump = df[df.trumppct > .5]
tm = trump.mean()
clinton = df[df.clintonpct > .5]
cl = clinton.mean()
 
#split into train and test
train = df.sample(frac=0.6, random_state=1)
test = df.loc[~df.index.isin(train.index)]
 
def Feature1():
#################################################################
#FEATURE 1 ---Use Recursive Feature Selection to Choose Variables
#################################################################
    print("Feature 1 - Recursive Feature Selection to Choose Variables")
 
    #Random Forest Algorithm
 
    #Results of ALgorithm 
    print("Feature 1 -- Variable Selection")
    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Num Features: %d" % rfecv.n_features_)
    print("Selected Features: %s" % rfecv.support_)
    print("Feature Ranking: %s" % rfecv.ranking_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
 
#Create a dataset that has the x variables for model selection
    print("The predictor variables are: ")
    print(new_predictor_columns)
     
def Feature2():
##############################
#FEATURE 2 EXPLORE THE DATASET
##############################
 
    #Find means for county facts for each of the candidates
 
    print("Feature 2 - Explore the Dataset with charts")
 
    #Join means and compare differences
    bars = pd.concat([tm, cl], axis=1)
    bars.columns = ['trump','clinton']
    bars = bars.drop(bars.index[[33,34,35,36,37]])
    bars['Ratio in Trump Win'] = bars.trump/bars.clinton
    bars['Ratio in Clinton Win'] = bars.clinton/bars.trump
     
    del bars['trump'] #remove columns not being used
    del bars['clinton']
    #remove vote totals
    bars2 = bars.sort_values(by='Ratio in Trump Win',ascending=False) #sort for best trump 
    trumpp = bars2.iloc[:5] #keep top 10
    del trumpp['Ratio in Clinton Win'] #hold onto just Trump data
 
##########MAKE SOME CHARTS###########
    #Make Trump Chart
    my_colors = ['r'] #Red for republican
     
    trumpp = trumpp.sort_values(by='Ratio in Trump Win')#resort for chart purpposes
    trumpp.plot(kind='barh', color=my_colors,legend=True,title="Top 5 Demographics of county Trump won", xlim=(0,2))
 
    #Make Clinton Chart
    bars2 = bars.sort_values(by='Ratio in Clinton Win',ascending=False)
    Clintonp = bars2.iloc[:5]
    del Clintonp['Ratio in Trump Win']
    my_colors = ['b'] #Blue for Democrats
    Clintonp = Clintonp.sort_values(by='Ratio in Clinton Win')
    Clintonp.plot(kind='barh', color=my_colors,legend=True,title="Top 5 Demographics of county Clinton won",xlim=(0,12))
 
    print("Trump won " + str(len(trump)) + " counties.")
    print("Clinton won " + str(len(clinton)) + " counties.")
 
 
#Run new random forest used predictors chosen in feature selection
 
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[new_predictor_columns], train["TrumpWin"])
predictions1 = rf.predict(test[new_predictor_columns]) 
logit = sm.Logit(train['TrumpWin'], train[new_predictor_columns])
result = logit.fit()
predictions = result.predict(test[new_predictor_columns])
pd.options.mode.chained_assignment = None  # removes default='warn' on assigning predictions into test dataframe  
logit_rmse = mean_squared_error(test["TrumpWin"], predictions)
rf_rmse = mean_squared_error(test["TrumpWin"], predictions1)
diff = logit_rmse-rf_rmse
if diff < 0:
    test.loc[:,'predictions'] = predictions
elif diff > 0:
    test.loc[:,'predictions'] = predictions1
def Feature3():
#################################
#FEATURE 3 MODEL SELECTION#######
#################################
 
 
 
 
 
    #Logistic Regression
    print("Feature 3 - Model Selection")
 
    #print (result.summary()) #Logistic Regression Summary 
    #odds = (np.exp(result.params)) 
    #print("\n"+"-"*50+"\n"+"Logistic Regression Odds Ratios"+"\n"+"-"*50)
    #print(round(odds,3)*100) #Odds ratio
 
    #Compare Models
     
    print("Root Mean Square Error Summary:")
    print("Logistic Regression RMSE " + str(logit_rmse))
    print("Random Forest RMSE: " + str(rf_rmse))
 
    if diff < 0:
        print("Logisitic Regression is the model chosen")
    elif diff > 0:
        print("Random Forest is the model chosen")
    print("\n")
 
d = test[test['TrumpWin'] == 1] #These are all the states that Trump won
e = d[d['predictions'] < .5] #and these are all the places where the prediction for Trump is less than 50%
 
def Feature4():
################################################
####FEATURE 4 WHERE DID HILLARY GO WRONG########
################################################
    print("Feature 4 - Where Did Hillary Go Wrong?")
    #Isolate battleground states
    #testbg = test[(test.state == 'CO') | (test.state == 'AZ') | (test.state == 'NV') | (test.state == 'NH')
    #               | (test.state == 'VA') | (test.state == 'FL')| (test.state == 'NC')| (test.state == 'PA')
    #               | (test.state == 'MI') | (test.state == 'WI') | (test.state == 'MN') | (test.state == 'OH')] 
 
 
 
  
    print("Trump won " + str(len(d))  + " counties in battleground states in the test dataset.")
    print("According to the algorithm, in counties Trump won, Clinton should have done better in at least " + str(len(e)) + " counties")
    print("Clinton missed an opportunity to win more votes in the following counties:")
    print(e.county + ", " + e.state)
    f = e.mean()
 
    miss = pd.concat([f, cl], axis=1) #make a table of the average clinton winning county and average where she "missed"
    miss.columns = ['clinton_miss','clinton_average']
    miss = miss.drop(miss.index[[0,9,10,32,35,36]])  #drop unnecessary data
    miss['diff'] = abs((miss.clinton_miss-miss.clinton_average)/miss.clinton_average) #compare the columns
    miss = miss.sort_values(by='diff',ascending=False)#resort for chart purpposes
 
    missp = miss
    print("Top 10 Leading Indicators of Where Clinton Missed")
    print(missp.iloc[:10])
 
def Feature5():
    #How about comparing to 2012 results?
    print("Feature 5 - Compare to 2012")
    df3 = pd.merge(e, df2, on=['county', 'state'])
    df3['obama_pct'] = df3.Obama/(df3.Obama+df3.Romney)
    df3['romney_pct'] = df3.Romney/(df3.Obama+df3.Romney)
     
    obama = 0
    clinton = 0
 
    for index, row in df3.iterrows():
        if ((row['obama_pct']) >  (row['clintonpct'])):
            obama += 1
        else:
            clinton += 1
 
    print ("\n" + "In these counties, in 2012, Obama did better than Clinton in 2016 in " + str(obama) + " counties. Clinton did better than Obama in " + str(clinton) + " counties.")
 
    df3['VoteDiff'] = (df3.obama_pct * (df3.clinton + df3.trump) - df3.clinton) 
    print("By State: " + "\n" + "="*30)
    new_votes = (df3.pivot_table('VoteDiff',index='state',aggfunc=sum).reset_index())
    print(new_votes.sort_values(by='VoteDiff',ascending=False))
 
 