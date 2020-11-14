import pandas as pandas, requests, json 
import numpy
from datetime import datetime, timezone
import time
from decimal import *
 

dataframe={'Longitude':[20,4,15], 'Latitude':[4,9,21],'RSSI':[20,10,8]}

def macSplit(singleMAC):  #Main data parsing function
    #singleMAC=byMAC.get_group(x[1])
    singleMAC=pandas.DataFrame(singleMAC)
    MACtime=singleMAC[['ts_sec', 'sensor_id']] #Select epoch timestamp and sensor id
    MACtime=MACtime.drop_duplicates()
    timeID=MACtime.pivot(index='ts_sec', columns='sensor_id', values='sensor_id') #Creates table of timestamps as indices and shows which 
    #sensors have data for that timestamp
    tColNames=list() #Holds list of column names
    for col in timeID.columns: #Creates columns to hold timestamp of last seen data of that sensor
        tColNames.append(col)
        colName=col+'t'
        timeID[colName]=''
    for y in tColNames: #Add data to the columns created above
        z=y+'t'
        for x in timeID.index.values:
            if pandas.isnull(timeID.loc[x].at[y]):
                timeID.loc[x].at[z]=timeID.iloc[timeID.index.get_loc(x)-1].at[z]
            else:
                timeID.loc[x].at[z]=x
#Above: If there is data at that time stamp, put that timestamp in the column
#If not, use timestamp of the cell above                
             
    size=int(len(timeID.columns.values)/2)  #get right half of data matrix
    cols=list(timeID.columns.values)[size:]
    for x in cols:  #Replace blanks with nan 
        timeID[x].replace('', numpy.nan, inplace=True)
    
    timeID.dropna(subset=cols, inplace=True) #Drop rows with nan in timestamp fields
    
    timeID=timeID[cols] #Next few lines: Filter out timestamps where the time window/difference is too high
    timeID['Diff']=timeID.max(axis=1)-timeID.min(axis=1)
    timeID=timeID.loc[timeID['Diff']<4]
    if(timeID.empty):
        return
    else:
        locations=pandas.DataFrame()
        bySensor=singleMAC.groupby('sensor_id')
        for x in timeID.index.values: #getting the data for each timestamp and applying algorithm
            vertices=pandas.DataFrame()        
            for y in cols:
                sensor=y[0:2] #subset str to get right columns
                sensordf=bySensor.get_group(sensor) #get the data for a sensor
                dataPoint=sensordf.iloc[(sensordf['ts_sec']-x).abs().argsort()[:1]] #Get data point closest to target timestamp
                #Takes difference of the target and sensor data, then sorts it and takes first one
                vertices=vertices.append(dataPoint) #Add it to a dataframe
            vertices['RSSI']=calculateDistance(vertices['signal']) #After adding, which should be equal to # of sensors
            timeMAC=vertices[['sourcemac', 'Timestamp']].head(n=1)
            MAC=timeMAC['sourcemac'] #get MAC and an avg timestamp for the data, to be added later
            Timestamp=timeMAC['Timestamp']  
            simplified=vertices[['lon','lat','RSSI']] #Subset correctly for algorithm
            simplified=simplified.rename(columns={"lon":"Longitude", "lat":"Latitude"})
            final=calculateEMinMax(simplified) #Algorithm application
            final=final.assign(MAC=pandas.Series(MAC.values))#Add columns for MAC and Timestamp
            final['MAC']=MAC.to_string(header=False, index=False) 
            final=final.assign(Timestamp=pandas.Series(Timestamp.values))
            pattern='%Y-%m-%d %H:%M:%S'
            final['epoch']=str(int(time.mktime(time.strptime(Timestamp.to_string(header=False,index=False) ,pattern))))
            locations=locations.append(final) #Add resulting to a final data frame for data return              
        minval=int(min(locations['epoch']))
        maxval=int(max(locations['epoch']))
        mintomax=pandas.Series(range(minval, maxval+1))
        mintomax=pandas.DataFrame(mintomax)
        mintomax=mintomax.rename(columns={0:"epoch"})
        mintomax['epoch']=mintomax['epoch'].astype(str)
        mintomax.index=mintomax['epoch']
        locationsnew=locations.drop_duplicates(subset='epoch', keep='first')
        locationsnew.index=locationsnew['epoch']
        joined=mintomax.join(locationsnew, lsuffix='_all', rsuffix='')
        joined=joined[['Longitude', 'Latitude', 'MAC', 'Timestamp', 'epoch']]
        for index in joined.index.values:
            if(pandas.isnull(joined.loc[index, 'Longitude'])):
                joined.loc[index]=joined.iloc[joined.index.get_loc(index)-1]
        joined['epoch']=joined.index.values
        joined=joined[['Longitude', 'Latitude', 'MAC', 'epoch']]
        return joined


def calculateDistance(rssi):
    x=(-40-rssi)/50
    meter=pow(10, x)/111015.5
    return meter

def calculateEMinMax(dataframe):
    #Block 1: Simple Min-Max localization algorithm
    length=len(dataframe)
    matrix=pandas.DataFrame(dataframe)
    matrix['sumx']=matrix['Longitude']+matrix['RSSI']
    matrix['sumy']=matrix['Latitude']+matrix['RSSI']
    matrix['diffx']=matrix['Longitude']-matrix['RSSI']
    matrix['diffy']=matrix['Latitude']-matrix['RSSI']
    xmax=matrix['diffx'].max()
    xmin=matrix['sumx'].min()
    ymax=matrix['diffy'].max()
    ymin=matrix['sumy'].min()
    points={'px':[xmax, xmax, xmin, xmin], 'py':[ymax,ymin,ymin,ymax]}
    dfpoints=pandas.DataFrame(data=points)
    #Block 2: Preparing data frame for merging to get all combinations of results of Block 1 and original vertices
    matrix=matrix.rename(columns={'Longitude':'x', 'Latitude':'y'})
    matrix['Sensor']=matrix['x'].astype(str)+' '+matrix['y'].astype(str)
    dfpoints['Vertex']=dfpoints['px'].astype(str)+' '+dfpoints['py'].astype(str)
    matrix['key']=1
    dfpoints['key']=1
    #Block 3: Outer merge, then compute manhattan distances and the square of difference between MD and original distances. Then compute the sum for every 
    #group of Min-Max boundary vertices, and take the inverse
    merged=matrix.merge(dfpoints, how='outer',on='key' )
    mergedFinal=merged[['x', 'y', 'px', 'py', 'RSSI']]
    mergedFinal.loc[:,"ManhattanD"]=abs(mergedFinal.loc[:,'x']-mergedFinal.loc[:,'px'])+abs(mergedFinal.loc[:,'y']-mergedFinal.loc[:,'py'])
    mergedFinal.loc[:,"SquareDiff"]=pow(mergedFinal.loc[:,'ManhattanD']-mergedFinal.loc[:,'RSSI'],2)
    mergedFinal=mergedFinal.sort_values(by=['px', 'py'])
    sums=mergedFinal['SquareDiff'].values.reshape(-1,length).sum(1)
    mergedFinal.loc[:,'Vertex']=mergedFinal.loc[:,'px'].astype(str)+' '+mergedFinal.loc[:,'py'].astype(str)
    vertices=mergedFinal.Vertex.unique()
    WeightCalcMatrix=pandas.DataFrame({'Vertex':vertices, 'SumDist':sums})
    WeightCalcMatrix['Inverse']=1/WeightCalcMatrix.SumDist
    new=WeightCalcMatrix.Vertex.str.split(" ", n=1, expand= True)
    #Block 4 Formatting and then compute the weighted centroid location
    WeightCalcMatrix['px']=new[0]
    WeightCalcMatrix['py']=new[1]
    WeightCalcMatrix['Fx']=WeightCalcMatrix.Inverse*WeightCalcMatrix.px.astype(float)
    WeightCalcMatrix['Fy']=WeightCalcMatrix.Inverse*WeightCalcMatrix.py.astype(float)
    x=sum(WeightCalcMatrix.Fx)/sum(WeightCalcMatrix.Inverse)
    y=sum(WeightCalcMatrix.Fy)/sum(WeightCalcMatrix.Inverse)
    #Block 5 #Formatting output
    x="{:.15f}".format(x)
    y="{:.15f}".format(y)
    xmax="{:.15f}".format(xmax)
    xmin="{:.15f}".format(xmin)
    ymin="{:.15f}".format(ymin)
    ymax="{:.15f}".format(ymax)
    finalPoints={'Longitude':[x], 'Latitude':[y]}
    #finalPoints={'Longitude':[xmax, xmax,xmin,xmin,x] , 'Latitude':[ymax,ymin,ymin,ymax,y]}

    point=pandas.DataFrame(data=finalPoints)
    return point

 
calculateDistance(-40)


 
csv=pandas.read_csv('C:/Users/jeffr/OneDrive/Documents/Python/es_data.csv')
data=csv[['ts_sec','lat', 'lon', 'signal', 'sensor_id', 'sourcemac', 'frame_type']]
data=data.dropna(thresh=5)
data['Timestamp']=data.ts_sec.apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
data['Timestamp']=pandas.to_datetime(data['Timestamp'])
byMAC=data.groupby('sourcemac')
x=data.sourcemac.unique()
finalData=pandas.DataFrame()
for y in x:
    z=byMAC.get_group(y)
    finalData=finalData.append(macSplit(z))
finalData=finalData.reset_index(drop=True)
finalData['epoch']=finalData['epoch']+"000"
finalData['Timestamp']=finalData.epoch.apply(lambda x: datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
counts=finalData.groupby('MAC').count()
highData=finalData.groupby('MAC').get_group(' F8:E4:FB:1B:0B:1F')
properties=['MAC', 'Timestamp']
geojson=df_to_geojson(highData, properties=properties, lat='Latitude', lon='Longitude')
geojson_str=json.dumps(geojson, indent=2)
output_filename = 'dataset.js'
with open(output_filename, 'w') as output_file:
    output_file.write('var dataset = {};'.format(geojson_str))


def df_to_geojson(df, properties, lat='latitude', lon='longitude'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [row[lon],row[lat]]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson
geojson=df_to_geojson(A6, properties=properties, lat='Latitude', lon='Longitude')
geojson_str=json.dumps(geojson, indent=2)
output_filename = 'dataset.js'
with open(output_filename, 'w') as output_file:
    output_file.write('var dataset = {};'.format(geojson_str))



 
# x=macSplit(A6)
# a=finalData.MAC.unique()



# point=pandas.DataFrame(columns=['Longitude','Latitude'])
# point=point.append({'Longitude':23, 'Latitude':23}, ignore_index=True)

# timestamp=timeID.index.values.tolist()
# pattern = '%Y-%m-%d %H:%M:%S'
# timestamp = list(map(lambda x: int(time.mktime(time.strptime(x, pattern))), timestamp))
# timeseries=pandas.Series(timestamp)


# def findClosest(senList, number):
#     vec=senList['ts_sec'].drop_duplicates()
#     senSort=senList.iloc[(vec-number).abs().argsort()[:1]]
#     return senSort

# def timeCheck(timestamp, splitSensor):
#     splitSensor=A6.groupby('sensor_id').apply(findClosest)
#     splitSensor['Dist']=calculateDistance(splitSensor['RSSI'])]
#     vertices=splitSensor[['Longitude', 'Latitude', 'Dist']]
#     finalVertices=calculateEMinMax(vertices)

# findClosest(A6s1, timetest)


minval=int(min(locations['epoch']))
maxval=int(max(locations['epoch']))
mintomax=pandas.Series(range(minval, maxval+1))
mintomax=pandas.DataFrame(mintomax)
mintomax=mintomax.rename(columns={0:"epoch"})
mintomax['epoch']=mintomax['epoch'].astype(str)
mintomax.index=mintomax['epoch']
locationsnew=locations.drop_duplicates(subset='epoch', keep='first')
locationsnew.index=locationsnew['epoch']
joined=mintomax.join(locationsnew, lsuffix='_all', rsuffix='')
joined=joined[['Longitude', 'Latitude', 'MAC', 'Timestamp', 'epoch']]
for index in joined.index.values:
    if(pandas.isnull(joined.loc[index, 'Longitude'])):
        joined.loc[index]=joined.iloc[joined.index.get_loc(index)-1]
joined['epoch']=joined.index.values
joined=joined[['Longitude', 'Latitude', 'MAC', 'epoch']]
        
        
pandas.isnull(joined.loc['1594650358', 'Longitude'])
joined.loc['1594650358']
        
        
 for y in tColNames: #Add data to the columns created above
        z=y+'t'
        for x in timeID.index.values:
            if pandas.isnull(timeID.loc[x].at[y]):
                timeID.loc[x].at[z]=timeID.iloc[timeID.index.get_loc(x)-1].at[z]
            else:
                timeID.loc[x].at[z]=x
#Above: If there is data at that time stamp, put

