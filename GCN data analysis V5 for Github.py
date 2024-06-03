# =============================================================================
# #STEP 1: Initial Data Exploration
# =============================================================================

#Dependencies
import os
import pandas as pd
import geopandas as gpd
import numpy as np

os.chdir() #removed for github
pond_data=gpd.read_file('Surveyed_priority_ponds_England_-5753754621454494271.geojson') #5430 rows

#remove extra columns and make a more and readable concise df
list(pond_data.columns)
pond_data=pond_data[['OBJECTID','Site_name','Survey_date','x_value','y_value','NS_com_ame',
                          'NEAREST_TOWN','NUTS212NM','geometry']].rename(columns={'Site_name':'Site','NEAREST_TOWN':'Nearest_town',
                          'NS_com_ame':'common_species_name','NUTS212NM':'County'},inplace=True)

#brief quality check 
pond_data.info() #some common names missing - that's expected as not every survey found a species
pond_data.dtypes
pond_data['Site'].unique() #very messy but not a huge deal as it's not part of the analysis (coords far more valuable)
pondpond_data['common_species_name'].unique() #'Great Crested Newt' and 'Common Toad, Great Crested Newt' - will need to rename latter
pond_data['common_species_name'] = pond_data['common_species_name'].replace(['Common Toad, Great Crested Newt'], 'Great Crested Newt')
pond_data['County'].unique() #"Wiltsire and Bristol/Bath area" isn't an appropriate group - many other examples, so I need to grab county via coordinates
pond_data['Nearest_town'].unique() #not familiar enough with towns of England - can possibly confirm spelling with a list of towns on a web site if these are needed for analysis 

#fixing county data as it's poor
from geopandas.tools import sjoin

county_lookup = gpd.read_file('./UK shapefile/UK trans.shp')
county_lookup = sjoin(pond_data, county_lookup, how='left',op="within").rename(columns={'NAME_2':'Extracted County'}, inplace=True)
county_lookup=county_lookup[['OBJECTID','NAME_1','Extracted County']]

#join extracted county df to pond_data df with extracted covariates, then clean up the columns
pond_data=pd.merge(pond_data,county_lookup,on='OBJECTID', how='left')
pond_data['County']=pond_data['Extracted County'].drop('Extracted County',axis=1)
del(county_lookup)

#replace the 2 county/country NAs
pond_data['County'] = np.where((pond_data['OBJECTID'] == 3451 | pond_data['OBJECTID'] == 3452), 'Merseyside', pond_data['County'])
pond_data['NAME_1'] = np.where((pond_data['OBJECTID'] == 3451| pond_data['OBJECTID'] == 3452), 'England', pond_data['NAME_1'])

#filter to England and save
pond_data=pond_data[pond_data['NAME_1'] == 'England'].drop('NAME_1',axis=1).to_csv('pond_data.csv',index=False)

# =============================================================================
# STEP 2A: clipping and reformating rasters to match data bounds and CRS
# =============================================================================

import fiona
import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

os.listdir('./tiffs/todo')

filename='GHI' #in future this can be a loop
imagery = rasterio.open('./tiffs/todo/'+filename+'.tif')
show(imagery) #looks fine
imagery.crs

# Transform projection of imagery to specific coordinate system. Specify output projection system
dst_crs = 'EPSG:27700'

# Input imagery file name before transformation
input_imagery_file = './tiffs/todo/'+filename+'.tif'
# Save output imagery file name after transformation
transformed_imagery_file = './tiffs/1 transformed/'+filename+'_trans.tif'

with rasterio.open(input_imagery_file) as imagery:
    transform, width, height = calculate_default_transform(imagery.crs, dst_crs, imagery.width, imagery.height, *imagery.bounds)
    kwargs = imagery.meta.copy()
    kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
    with rasterio.open(transformed_imagery_file, 'w', **kwargs) as dst:
        for i in range(1, imagery.count + 1):
            reproject(
                source=rasterio.band(imagery, i),
                destination=rasterio.band(dst, i),
                src_transform=imagery.transform,
                src_crs=imagery.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
del(imagery,kwargs,i,height,dst,transform,transformed_imagery_file,width,dst_crs,input_imagery_file)  

# Read Shape file
with fiona.open('./UK shapefile/UK trans.shp', "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

# read imagery file
with rasterio.open('./tiffs/1 transformed/'+filename+'_trans.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

# Save clipped imagery
out_meta.update({"driver": "GTiff",
                  "height": out_image.shape[1],
                  "width": out_image.shape[2],
                  "transform": out_transform})

with rasterio.open("./tiffs/2 clipped/"+filename+"_trans_clip.tif", "w", **out_meta) as dest:
    dest.write(out_image)
del(src,shapes,tr_imagery,out_image,out_meta,dest,out_transform,shapefile)

# =============================================================================
# # STEP 2B: Extract covariates per county (for use in missing site data)
# =============================================================================

import numpy as np
import rasterstats as rs
import rioxarray as rxr
import earthpy.plot as ep

#get list of files in clipped folder for buffering
tiff_names=[]
for file in os.listdir('./tiffs/2 clipped'):
     filename = os.fsdecode(file)
     tiff_names.append(filename)
del(file,filename)

world = gpd.read_file('./UK shapefile/UK trans.shp') #this is a transformed file (using the method above) that I saved seperately using world.to_file('./UK shapefile/UK trans.shp', driver='ESRI Shapefile') 
world=world[['NAME_2','geometry']]

#create a list of variable and measure names
variable_names=['elevation','GHI','hfp','hillshade','population','precipitation','windspeed10']
measure_names=['_count' ,'_min', '_mean', '_max', '_median']
colnames=[]

#loop through each variable and add _count _min _mean _max _median so that we get variable_count,variable_min etc.
for i in variable_names:
    for j in measure_names:
        column=i+j
        colnames.append(column)
del(i,j,column,measure_names,variable_names)

county_df=pd.DataFrame()
for i in tiff_names:
    tiff_data=rxr.open_rasterio("./tiffs/2 clipped/"+i, masked=True).squeeze()
    tiff_data_no_zeros = tiff_data.where(tiff_data > 0, np.nan)
    del(tiff_data)
    tiff_data_stats = rs.zonal_stats(world, #get zonal stats
                                       tiff_data_no_zeros.values,
                                       nodata=-999,
                                       affine=tiff_data_no_zeros.rio.transform(),
                                       geojson_out=True,
                                       copy_properties=True,
                                       stats="count min mean max median")
    tiff_data_buffered = gpd.GeoDataFrame.from_features(tiff_data_stats) # Turn extracted data into a pandas geodataframe
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.bar(tiff_data_buffered['NAME_2'],
           tiff_data_buffered['median'],
           color="purple")
    ax.set(title=i)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
    del(ax,fig)
    print(i+" has: "+str(tiff_data_buffered['median'].isna().sum()) + " NAs")
    tiff_data_buffered_conc=tiff_data_buffered[['count', 'min', 'mean', 'max' ,'median']]
    county_df=pd.concat([county_df,tiff_data_buffered_conc],axis=1) 
county_df.columns=colnames
del(colnames,tiff_data_stats,tiff_data_no_zeros,tiff_data_buffered,tiff_data_buffered_conc,i)

#replace NA with column mean
for column in county_df:
        mean_value=county_df[column].mean()
        county_df[column].fillna(value=mean_value, inplace=True)
del(column,mean_value)

#save df 
county_extracted_covariates =pd.concat([world,county_df],axis=1).rename(columns={'NAME_2':'County'}, inplace=True).to_csv('county_extracted_covariates_mean_imputation.csv',index=False)
del(county_df,tiff_names,world)

# =============================================================================
# #STEP 2C Create a buffer around pond coordinates and extract covariates
# =============================================================================

#Precipitation
#https://www.arcgis.com/home/item.html?id=e6ab693056a9465cbc3b26414f0ddd2c
#average for the year 2000 at 5km resolution

#Population
#https://data.humdata.org/dataset/worldpop-population-density-for-united-kingdom-of-great-britain-and-northern-ireland/resource/634e1307-8134-4da9-b627-27c09a3fd26d
#average for 2020 at a 1km resolution

#solar Global Horizontal Irradiation (GHI)
#https://solargis.com/maps-and-gis-data/tech-specs
# yearly total [kWh/m2] at 250m resolution for 2020

#elevation/hillshade
#https://www.eea.europa.eu/data-and-maps/data/digital-elevation-model-of-europe
#2004, resolution at 1km

#windspeed
#http://www.meiotic.co.uk/my/research/uk-wind-data/
#mean annual (unknown year) 10m above the ground at 1km resolution

#HFP
#https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint
#The 2009 Human Footprint, 2018 Release provides a global map of the cumulative human pressure on the environment in 2009, at a spatial resolution of ~1 km. The human pressure is measured using eight variables including built-up environments, population density, electric power infrastructure, crop lands, pasture lands, roads, railways, and navigable waterways. The data set is produced by Venter et.al., and is available in the Mollweide projection.

#https://www.earthdatascience.org/courses/use-data-open-source-python/spatial-data-applications/lidar-remote-sensing-uncertainty/extract-data-from-raster/
#awesome buffer tutorial!

#get list of files in clipped folder for buffering
tiff_names=[]
for file in os.listdir('./tiffs/2 clipped'):
     filename = os.fsdecode(file)
     tiff_names.append(filename)
del(file,filename)

#check data is in correct format :
for i in tiff_names:
    print(i)
    tiff_data=rxr.open_rasterio("./tiffs/2 clipped/"+i, masked=True).squeeze()
    print(tiff_data.rio.nodata)
    print(tiff_data.rio.bounds()) 
    print(tiff_data.rio.crs) #should all be 27700
    print(tiff_data.rio.nodata)
    print(tiff_data.shape) 
    print(tiff_data.rio.resolution())
    tiff_data.plot()    #geography sense check to see if values exist and if the clipping script borked somehow
    plt.xlim([130000, 700000])
    plt.ylim([0,670000])
    plt.title("Distribution of "+i+" pixel values")
    plt.savefig('./visualisations/Visual_distribution_of_'+i+'_pixel_values.png', bbox_inches='tight')  
    plt.show()
    ax = ep.hist(tiff_data.values,  #another sense check
                 figsize=(8, 8),
                 colors="purple",
                 ylabel="Total Pixels",
                 title="Distribution of "+i+" pixel values")
    plt.savefig('./visualisations/Bar_distribution_of_'+i+'_pixel_values.png', bbox_inches='tight')  
    plt.show()
del(ax,i) 

#plot points over raster just to confirm CRS matches and that no coordinates are outside 
for i in tiff_names:
    print(i)
    tiff_data=rxr.open_rasterio("./tiffs/2 clipped/"+i, masked=True).squeeze()
    fig, ax = plt.subplots(figsize=(10,12))
    base = tiff_data.plot()
    pond_data.plot(ax=ax)
    plt.xlim([130000, 700000])
    plt.ylim([0,670000])
    plt.show()
del(ax,base,fig,i)

#create a list of variable and measure names
variable_names=['elevation','GHI','hfp','hillshade','population','precipitation','windspeed10']
measure_names=['_count' ,'_min', '_mean', '_max', '_median']
colnames=[]

#loop through each variable and add _count _min _mean _max _median so that we get variable_count,variable_min etc.
for i in variable_names:
    for j in measure_names:
        column=i+j
        colnames.append(column)
del(i,j,column,measure_names,variable_names)

# Create a buffered polygon layer from your plot location points
pond_data_poly = pond_data.copy()
pond_data_poly["geometry"] = pond_data.geometry.buffer(2500) #GCN home range can be 500m away from ponds

#loop through each tiff file and extract stats
extract_df=pd.DataFrame()
for i in tiff_names:
    tiff_data=rxr.open_rasterio("./tiffs/2 clipped/"+i, masked=True).squeeze()
    tiff_data_no_zeros = tiff_data.where(tiff_data > 0, np.nan)
    del(tiff_data)
    tiff_data_stats = rs.zonal_stats(pond_data_poly, #get zonal stats
                                       tiff_data_no_zeros.values,
                                       nodata=-999,
                                       affine=tiff_data_no_zeros.rio.transform(),
                                       geojson_out=True,
                                       copy_properties=True,
                                       stats="count min mean max median")
    tiff_data_buffered = gpd.GeoDataFrame.from_features(tiff_data_stats) # Turn extracted data into a pandas geodataframe
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.bar(tiff_data_buffered['Site'],
           tiff_data_buffered['median'],
           color="purple")
    ax.set(title=i)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
    del(ax,fig)
    print(i+" has: "+str(tiff_data_buffered['median'].isna().sum()) + " NAs")
    tiff_data_buffered_conc=tiff_data_buffered[['count', 'min', 'mean', 'max' ,'median']]
    extract_df=pd.concat([extract_df,tiff_data_buffered_conc],axis=1) 
del(tiff_data_stats,tiff_data_no_zeros,pond_data_poly,tiff_data_buffered,tiff_data_buffered_conc,i)
extract_df.columns=colnames

#save df 
pond_data_with_extracted_covariates =pd.concat([pond_data,extract_df],axis=1) 
#pond_data_with_extracted_covariates=pd.merge(pond_data,joined_df,on='OBJECTID', how='left')
pond_data_with_extracted_covariates.to_csv('extract_df.csv',index=False)
del(extract_df)

# =============================================================================
# #STEP 2D: replace missing site data using county data
# =============================================================================

#load up county fix
county_covs=pd.read_csv('county_extracted_covariates_mean_imputation.csv')

#check number of NAs in site level data
for column in pond_data:
    if (pond_data[column].isna().sum() >0):
        print(column+" has: "+str(pond_data[column].isna().sum()) + " NAs")
del(column)

#for each tiff, create a set of columns to create a sub df with
replaced_df=pond_data[['OBJECTID']]
names=['elevation','GHI','hfp','hillshade','population','precipitation','windspeed10']
#names=['hfp'] #debugging
for k in names:
    variable_names=k
    measure_names=['_count' ,'_min', '_mean', '_max', '_median']
    colnames=[]
    colnames.extend(('OBJECTID','County'))
    for j in measure_names:
            column=k+j
            colnames.append(column)
    del(j,column,measure_names,variable_names)
    #if the count variable for any tiff is 0, the proceeding stats will always be NAs
    column=k+'_count'
    del(k)
    #subset df
    pond_data_subdf=pond_data[colnames]
    #filter to 0/NAs
    pond_data_subdf=pond_data_subdf[pond_data_subdf[column]==0]
    pond_data_subdf=pond_data_subdf[['OBJECTID','County']] #will get the columns from the county df
    #subset the pond_data df to the relevant columns and filter out the empty rows
    non_na_subset=pond_data[colnames]
    non_na_subset=non_na_subset.loc[non_na_subset[column] != 0]
    #subset county df as I don't want to join the whole thing, then join
    colnames.remove('OBJECTID') #No object ID but all the other columns are the same
    county_covs_subdf=county_covs[colnames]
    joined_df=pd.merge(pond_data_subdf,county_covs_subdf,on='County', how='left') 
    del(column,colnames)
    #check row count tallies
    if len(pond_data.index)-(len(non_na_subset.index)+len(joined_df.index)) !=0:
        print('row counts dont tally')
    else:
        print('row counts tally')
    joined_df=joined_df.append(non_na_subset)
    #check for duplicate rows
    joined_df.duplicated(subset=['OBJECTID']).any()
    del(county_covs_subdf,pond_data_subdf)
    joined_df = joined_df.drop('County',axis=1)
    replaced_df=pd.merge(replaced_df,joined_df,on='OBJECTID', how='left') 
del(names,non_na_subset,joined_df)

#tidy up the dataframes and merge the replaced data with the original dataset
variable_names=['elevation','GHI','hfp','hillshade','population','precipitation','windspeed10']
measure_names=['_count' ,'_min', '_mean', '_max', '_median']
colnames=[]
for i in variable_names:
    for j in measure_names:
        column=i+j
        colnames.append(column)
del(i,j,column,measure_names,variable_names)
pond_not_replaced=pond_data.drop(colnames,axis=1)
whole_replaced_df=pd.merge(replaced_df,pond_not_replaced,on='OBJECTID', how='left') 
del(colnames,pond_not_replaced,replaced_df)

#some tests to compare values before and after replacing to see if there's any screw ups
new_stat=whole_replaced_df.describe()
old_stat=pond_data.describe() #medians are basically the same!
del(new_stat,old_stat)

#plot new/old median for the different variables
import matplotlib.pyplot as plt
plot_variables=['elevation_median','GHI_median','hfp_median','hillshade_median','population_median','precipitation_median','windspeed10_median']
for i in plot_variables:
    fig, ax = plt.subplots(figsize=(10,12))
    base = whole_replaced_df[i].plot(kind='hist', edgecolor='black')
    pond_data[i].plot(kind='hist', edgecolor='black')
    plt.title(i)
    plt.show()
del(ax,base,fig,i,plot_variables)

reordered=whole_replaced_df[pond_data.columns]
reordered.to_csv('pond_data_no_missing_data.csv',index=False)

# =============================================================================
# #Step 3A visualising spatial relationships for diagnostic analysis
# =============================================================================

from shapely import wkt
import folium
import webbrowser

#load covariate csv at site granularity
pond_data=pd.read_csv('pond_data_no_missing_data.csv')
#load up county fix
county_data=pd.read_csv('county_extracted_covariates_mean_imputation.csv')

#read them as geodataframes - make chloropleth using county df, and plot pond_data sites on top
pond_data['geometry'] = pond_data['geometry'].apply(wkt.loads)
pond_data = gpd.GeoDataFrame(pond_data, crs='epsg:27700')
county_data['geometry'] = county_data['geometry'].apply(wkt.loads)
county_data = gpd.GeoDataFrame(county_data, crs='epsg:27700')

#recode GCN to 1, other species or no species to 0
pond_data['count'] = pond_data['common_species_name'].map(lambda x: 1 if x == 'Great Crested Newt' else 0)

#get GCN sums per county
sums=pd.DataFrame(pond_data.groupby(['County'])['count'].sum())
sums = sums.reset_index(level=0)

#do a count for counties and see if there's duplicates in county data
county_data=pd.merge(county_data,sums,on='County', how='left') 
county_data['count'] = county_data['count'].fillna(0)
county_data.to_csv('County_data_summed.csv',index=False)

#investigate row mismatch
county_data_0s=county_data[county_data['count'] <1]
county_data_nulls=county_data[county_data['count'].isna()]
del(county_data_0s,county_data_nulls,sums)

#plot chloropleth with pond sites for a sense check - confirmed some counties have 0
fig, ax = plt.subplots(figsize=(10,12))
cmap='Greys' #don't want to use reds or greens due to prevalence of rg colourblind in men
base =county_data.plot(column='count',ax=ax, cmap=cmap, edgecolor='black',linewidth=0.5, legend=True)
pond_data[pond_data['common_species_name'] == "Great Crested Newt"].plot(ax=base, markersize=8,color="gold",marker="o",edgecolor = 'black',linewidth = 0.05, label="Occupied") 
pond_data[pond_data['common_species_name'] != "Great Crested Newt"].plot(ax=base, markersize=2,color="blue",marker="o",label="Absent") 
plt.suptitle('Which counties have the most surveyed ponds \n and total GCN counts?', fontsize=20,y=0.91)
plt.xlim([130000, 700000])
plt.ylim([0,670000])
fig.tight_layout()
plt.legend(loc="upper right")    
plt.savefig('./visualisations/GCN_survey_chloropleth.png', bbox_inches='tight')
plt.show()
del(ax,fig,base,cmap)

#interactive map
f = folium.Map(location=[52.8225, -0.776], zoom_start=7,dragging=False)
m = county_data.explore(
     m=f,
     column="count",  # make choropleth based on "BoroName" column
     tooltip=["County",'count'],
     legend=True, # show legend
     k=10, # use 10 bins?
     legend_kwds=dict(colorbar=False), # do not use colorbar?
     name="County" # name of the layer in the map
)

pond_data[pond_data['common_species_name'] != "Great Crested Newt"].explore(
     m=m, # pass the map object
     color="blue", # use red color on all points
     marker_kwds=dict(radius=1, fill=True), # make marker radius 10px with fill
     tooltip="Site", # show "name" column in the tooltip
     tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
)

pond_data[pond_data['common_species_name'] == "Great Crested Newt"].explore(
     m=m, # pass the map object
     color="yellow", # use red color on all points
     marker_kwds=dict(radius=1, fill=True), # make marker radius 10px with fill
     tooltip="Site", # show "name" column in the tooltip
     tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
)

m.save("GCN_survey_chloropleth_interactive.html") # show map
webbrowser.open("GCN_survey_chloropleth_interactive.html")
del(m)

# =============================================================================
# # STEP 4: Determine best model architecture and variables to predict GCN counts
# =============================================================================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, plot_tree
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

os.chdir() #removed for github
pond_data=pd.read_csv('pond_data_no_missing_data.csv')
nulldate=pond_data[pond_data['Survey_date'].isna()] #none contain newts it seems. only 200 rows... remove?
del(nulldate)
pond_data['GCN_present'] = pond_data['common_species_name'].map(lambda x: 1 if x == 'Great Crested Newt' else 0)
pond_data = pond_data.drop(['Site','Nearest_town','County','Survey_date', 'x_value', 'y_value', 'common_species_name','geometry', 'month_name'],axis=1)
pond_data=pond_data.set_index('OBJECTID')

#only keep 'median' labels
filterlist=[]
variables=[]
pos = 0 #first columns arent relevant for this
end=len(pond_data.columns) #neither is the last one
while pos<end:
    colname = pond_data.columns[pos]
    pos=pos+1
    variables.append(colname)
del(pos,colname,end)
variables=pd.Series(variables) #turn variables vector into series for the below to work
typestoremove=['_count','_min','_max','_mean']
for i in typestoremove:
    delstring=i
    filters=variables[variables.str.endswith(delstring)] #get all columns that end in '_min' for example
    filterlist.append(filters)
for i in filterlist:
    filters=i
    variables = [s for s in variables if not any(f in s for f in filters)] #remove everything in list above
del(filters,i,delstring,typestoremove,filterlist)
pond_data=pond_data[variables]

#define features (and scale to allow for easier comparison)
X=pond_data[variables]  # Features
del(variables)
X=X.drop('GCN_present',axis=1)
xtoscale=X[['elevation_median','GHI_median','hfp_median','hillshade_median','population_median','precipitation_median','windspeed10_median']]
xnottoscale=X[['month','year','day','day_in_week']]
scaler = StandardScaler()
standardized_df = scaler.fit_transform(xtoscale)
standardized_df = pd.DataFrame(standardized_df, columns=xtoscale.columns)
standardized_df.index=xtoscale.index
del(xtoscale)
standardized_df=pd.merge(xnottoscale,standardized_df,on='OBJECTID', how='left') 
del(xnottoscale)
X=standardized_df
del(standardized_df,scaler)

#check for NAs - shouldn't be any due to earlier code
for column in X:
    if (X[column].isna().sum() >0):
        print(column+" has: "+str(X[column].isna().sum()) + " NAs")
del(column)

#remove date variables from X as they're not relevant to *where* GCNs are found
X=X.drop(['year','month','day_in_week','day'],axis=1)
                             
#define label
y=pond_data['GCN_present']

#create combinations of variables to test and see how accuracy improves
Xcopy=X
col_names=['cols','modeltype','accuracy']
combi_var_acc=pd.DataFrame(columns=col_names)
del(col_names)
input = list(X)
output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])
output.pop(0)

#start loop of iterative variables here
dfs = []
models = [
('LogReg', LogisticRegression()),
('RF', RandomForestClassifier()),
('KNN', KNeighborsClassifier()),
('SVM', SVC()),
('GNB', GaussianNB()),
('XGB', XGBClassifier())
]
results = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted',
'f1_weighted', 'roc_auc']
target_names = ['absent', 'present']
modelid=1
for i in output:
    X=Xcopy[i]
    y=pond_data['GCN_present'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True,random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train,y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cols=', '.join(i)
        accuracy=metrics.accuracy_score(y_test, y_pred)
        modeltype=name
        combi_var_acc.loc[len(combi_var_acc)] = [cols,modeltype,accuracy]
        print(name)
        print(i)
        print(classification_report(y_test, y_pred,target_names=target_names))
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        this_df['variables'] = ', '.join(i)
        this_df['modelcombo'] = name + ' ' + ', '.join(i)
        this_df['modelid'] = str(modelid)
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
        modelid=modelid+1
combi_var_acc=(combi_var_acc.sort_values('accuracy',ascending=False))
del(modelid,pond_data,Xcopy,modeltype,this_df,accuracy,cols,y_pred,X_train,X_test,y_train,y_test,clf,input,output,i,model,models,cv_results,name,names,kfold,dfs,target_names,scoring,results)  

#save because this took hours
final=pd.read_csv('ML_model_stats.csv')
#final.to_csv('ML_model_stats_backup.csv',index=False) #making a copy just in case I overwrite this

#get the 5 best models and their model id
final=(final.sort_values('test_accuracy',ascending=False))
finalbest=final.head(5)
finalbest=finalbest['modelid']
filter=list()
for i in finalbest:
    print(i)
    filter.append(i)
finalbest=final[final['modelid'].isin(filter)]
del(i,filter)
final=finalbest
del(finalbest)

#bootstrap
bootstraps = []
for modelid in list(set(final.modelid.values)):
    model_df = final.loc[final.modelid == modelid]
    bootstrap = model_df.sample(n=300, replace=True)
    bootstraps.append(bootstrap)
bootstrap_df = pd.concat(bootstraps, ignore_index=True)
results_long = pd.melt(bootstrap_df,id_vars=['modelid'],var_name='metrics', value_name='values')
time_metrics = ['fit_time','score_time'] 

## PERFORMANCE METRICS
results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # getdf without fit data
results_long_nofit=results_long_nofit[results_long_nofit['metrics'] != 'variables']
results_long_nofit=results_long_nofit[results_long_nofit['metrics'] != 'model']
results_long_nofit=results_long_nofit[results_long_nofit['metrics'] != 'modelid'] #needed?
results_long_nofit=results_long_nofit[results_long_nofit['metrics'] != 'modelcombo']
results_long_nofit = results_long_nofit.sort_values(by='values')
del(bootstraps,bootstrap,modelid,model_df)

## TIME METRICS
results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
results_long_fit = results_long_fit.sort_values(by='values')
del(results_long)

#plot the result of that
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="modelid", y="values", hue="metrics",data=results_long_nofit, palette="Set3")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Classification Metric')
plt.show()
plt.savefig('benchmark_models_performance.png',bbox_inches = 'tight')
del(g)

#plot train and score times
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="modelid", y="values", hue="metrics",
data=results_long_fit, palette="Set3")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Fit and Score Time')
plt.savefig('./benchmark_models_time.png',dpi=300)
#del(g)

#comparisons to see if models statisitcally overlap/differ when having difficulty choosing
metrics = list(set(results_long_nofit.metrics.values))
comparisons=bootstrap_df.groupby(['modelid'])[metrics].agg([np.std, np.mean])
del(results_long_nofit,comparisons)
time_metrics = list(set(results_long_fit.metrics.values))
bootstrap_df.groupby(['model'])[time_metrics].agg([np.std, np.mean])
del(time_metrics,bootstrap_df)

# =============================================================================
# #STEP 4B: Tune best model architecture and variable comination
# =============================================================================

#grab best ID, and the associated model architecture and formula from above step
final=pd.read_csv('ML_model_stats.csv')
bestid=732 #based on manual choice - there was lots of overlap and *some* performances were better/worse
bestmodel=final[final['modelid']==bestid].iat[0,7]
bestformula=final[final['modelid']==bestid].iat[0,8]
del(final)
print(bestmodel + ' - ' + bestformula)

#filter the dataframe by the formula
bestformula=bestformula.split(",")
bestformula = [x.strip(' ') for x in bestformula]
X_best=X[bestformula]
del(bestmodel,bestformula,X)

#train model
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3,random_state=9090,shuffle=True)
#evalset = [(X_train, y_train), (X_test,y_test)]

#parameter tuning - 7 each? this took 2 hours so reduce that next time
iterations=[100,50,200,500,3000,5000,10000]
learnrate=[0.3,0.2,0.1,0.5,0.1,0.05,0.01]
subbing=[1,0.75,0.5,0.3,0.1,0.05,0.01]
sampling=[1,0.75,0.5,0.3,0.1,0.05,0.01]

#metrics to save to work out which is the best tuning
cols= ['accuracy','n_estimators','eta','subsample','colsample_bytree']
trainingparameters=pd.DataFrame(columns=cols)
del(cols)

for i in iterations:
    for j in learnrate:
        for k in subbing:
            for l in sampling:
                clf = XGBClassifier(n_estimators=i,eta=j,subsample=k,colsample_bytree=l).fit(X_train, y_train)#, eval_metric='logloss', eval_set=evalset)
                y_pred = clf.predict(X_test)
                accuracy=metrics.accuracy_score(y_test, y_pred)
                # results = clf.evals_result()
                # plt.plot(results['validation_0']['logloss'], label='train')
                # plt.plot(results['validation_1']['logloss'], label='test')
                # plt.legend()
                # plt.show()
                print ('n_iterations= '+ str(i) + 'learn rate= ' + str(j)+ 'subsampling= ' + str(k) + 'colsampling by tree= ' + str(l))
                trainingparameters.loc[len(trainingparameters)] = [accuracy,i,j,k,l]
trainingparameters.to_csv('best parameters for model 732.csv',index=False)
del(i,j,k,l,iterations,accuracy,learnrate,sampling,subbing,clf,X_test,X_train,y_pred,y_test,y_train)

#grab best parameters
trainingparameters=(trainingparameters.sort_values('accuracy',ascending=False))
print('The highest accuracy acheived was ' + str(round(trainingparameters.iat[0,0]*100,2))+'% with:\n ' 
      + str(trainingparameters.iat[0,1]) + ' iterations \n '
      + str(trainingparameters.iat[0,2]) +' eta \n ' + str(trainingparameters.iat[0,3]) + ' subsampling rate, and \n '
      + str(trainingparameters.iat[0,4]) + ' column sampling rate')
iterations=trainingparameters.iloc[0,1]
learnrate=trainingparameters.iloc[0,2]
subbing=trainingparameters.iat[0,3]
sampling=trainingparameters.iat[0,4]

#retrain using best parameters
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3,random_state=9090,shuffle=True)
evalset = [(X_train, y_train), (X_test,y_test)]
clf = XGBClassifier(n_estimators=50,eta=0.05,subsample=0.75,colsample_bytree=0.5).fit(X_train, y_train, eval_metric='logloss', eval_set=evalset)
y_pred = pd.DataFrame(clf.predict(X_best),index=X_best.index) #predict with entire dataset
accuracy=metrics.accuracy_score(y, y_pred)
accuracy=round(accuracy*100,1)

#plot FIV out of curiousity
feature_imp = pd.Series(clf.feature_importances_,index=X_best.columns).sort_values(ascending=False)
fig,ax = plt.subplots(1)
sns.barplot(x=feature_imp, y=feature_imp.index)
label = ['Human footprint','Total solar radiation', 'Precipitation' ,'Windspeed', 'Elevation', 'Hillshade'] # I don't like being this manual, but this is for a one off visualisation - in future I need to come back and dynamically strip/rename
ax.set_yticklabels(label)  
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Which features were most important for model " + str(bestid) + " \n to produce " + str(accuracy)+"% accuracy?",fontsize=15,y=1.03,x=.35)
plt.legend()
plt.savefig('./visualisations/FIV.png', bbox_inches='tight')  
plt.show()

#visualising example decision tree
dump_list = clf.get_booster().get_dump() 
num_trees = len(dump_list) #grab number of trees

plt.figure(figsize=(50,50))  # set plot size (denoted in inches)
plot_tree(clf,num_trees=0, rankdir='LR')
plt.rcParams['figure.dpi'] = 500
plt.savefig('./visualisations/tree.jpeg', bbox_inches='tight')  
#plt.gcf().set_size_inches(18.5, 10.5)
plt.show()

#merge X_best for index, y, and y_pred
PredvsAct=pd.merge(y_pred,y, left_index=True, right_index=True)
PredvsAct_covs=pd.merge(X_best,PredvsAct, left_index=True, right_index=True)
del(PredvsAct)

#load in pond_data again and join columns to make a predicted vs actual df with all data
pond_data=pd.read_csv('pond_data_no_missing_data.csv')
pond_data_droppedvalues = pond_data[['OBJECTID','Site','Nearest_town','County','month','year','day','day_in_week','geometry']]
pond_data_droppedvalues=pond_data_droppedvalues.set_index('OBJECTID')
PredvsAct_all_covs=pd.merge(pond_data_droppedvalues,PredvsAct_covs, left_index=True, right_index=True)
PredvsAct_all_covs.rename(columns={0:'Pred_GCN_present'},inplace=True)
PredvsAct_all_covs.to_csv('PredvsAct_all_covs.csv',index=False)
del(pond_data,pond_data_droppedvalues,PredvsAct_covs)

# =============================================================================
# #STEP 4C: Use best tuned model to predict GCN counts
# =============================================================================

#load predicted vs actual df created in previous step
PredvsAct_all_covs=pd.read_csv('PredvsAct_all_covs.csv')

#sum by county
Act_county=PredvsAct_all_covs.groupby(['County'])['GCN_present'].sum()
Pred_county=PredvsAct_all_covs.groupby(['County'])['Pred_GCN_present'].sum()
PredvsAct_county=pd.merge(Act_county,Pred_county, on='County')
PredvsAct_county=(PredvsAct_county.sort_values('GCN_present',ascending=False))
del(Act_county,Pred_county)

#plot relationship
fig,ax = plt.subplots(1)
predicted_counts=PredvsAct_county['Pred_GCN_present']
actual_counts = PredvsAct_county['GCN_present']
fig.suptitle('Predicted versus actual GCN counts at a county level',y=0.94)
predicted, = plt.plot(PredvsAct_county.index, predicted_counts, 'go-', label='Predicted counts')
actual, = plt.plot(PredvsAct_county.index, actual_counts, 'ro-', label='Actual counts')
ax.set_xticklabels([])
ax.set_xticks([])
plt.legend(handles=[predicted, actual])
fig.savefig('./visualisations/PredvsAct_linegraph.png', bbox_inches='tight')
plt.show()
del(fig,ax,actual_counts,predicted_counts,predicted,actual)

#load county df from step 3A, merge new predicted counts in, and calculate differences as well as only positives
county_data=pd.read_csv('County_data_summed.csv')
county_data=county_data.set_index('County')
merged_county_data=pd.merge(PredvsAct_county,county_data,left_index=True, right_index=True)
merged_county_data['difference']=merged_county_data['Pred_GCN_present']-merged_county_data['GCN_present']
merged_county_data['increases']=merged_county_data['difference']
merged_county_data['increases'] = merged_county_data['increases'].clip(lower=0)
merged_county_data.to_csv('Merged_county_data_summed.csv',index=False)

#plot new data
merged_county_data['geometry'] = merged_county_data['geometry'].apply(wkt.loads)
merged_county_data = gpd.GeoDataFrame(merged_county_data, crs='epsg:27700')

fig, ax = plt.subplots(figsize=(10,12))
cmap='Greys' #don't want to use reds or greens due to prevalence of rg colourblind in men
base =merged_county_data.plot(column='GCN_present',ax=ax, edgecolor='black',linewidth=0.5, cmap=cmap,legend=True)#, edgecolor='black',linewidth=0.5)
plt.suptitle('Actual GCN counts across all years per county', fontsize=22,y=0.91,x=0.45)
plt.xlim([130000, 700000])
plt.ylim([0,670000])
fig.tight_layout()
plt.legend(loc="upper right")  
plt.savefig('./visualisations/GCN_survey_chloropleth_noponds.png', bbox_inches='tight')  
plt.show()
del(ax,fig,base,cmap)

fig, ax = plt.subplots(figsize=(10,12))
cmap='Greys' #don't want to use reds or greens due to prevalence of rg colourblind in men
base =merged_county_data.plot(column='Pred_GCN_present',ax=ax, edgecolor='black',linewidth=0.5, cmap=cmap,legend=True)#, edgecolor='black',linewidth=0.5)
plt.suptitle('Predicted GCN counts across all years \n per county', fontsize=25,y=0.91,x=0.45)
plt.xlim([130000, 700000])
plt.ylim([0,670000])
fig.tight_layout()
plt.legend(loc="upper right")   
plt.savefig('./visualisations/GCN_survey_chloropleth_noponds_predicted.png', bbox_inches='tight')  
plt.show()
del(ax,fig,base,cmap)

fig, ax = plt.subplots(figsize=(10,12))
cmap='Greys' #don't want to use reds or greens due to prevalence of rg colourblind in men
base =merged_county_data.plot(column='difference',ax=ax, edgecolor='black',linewidth=0.5, cmap=cmap, legend=True)#, edgecolor='black',linewidth=0.5)
plt.suptitle('Difference between Predicted and Actual \n GCN counts across all years per county', fontsize=25,y=0.91,x=0.45)
plt.xlim([130000, 700000])
plt.ylim([0,670000])
fig.tight_layout()
plt.legend(loc="upper right")   
plt.savefig('./visualisations/GCN_survey_chloropleth_noponds_differences.png', bbox_inches='tight')  
plt.show()
del(ax,fig,base,cmap)

fig, ax = plt.subplots(figsize=(10,12))
cmap='Greys' #don't want to use reds or greens due to prevalence of rg colourblind in men
base =merged_county_data.plot(column='increases',ax=ax,edgecolor='black',linewidth=0.5, cmap=cmap, legend=True)#, edgecolor='black',linewidth=0.5)
plt.suptitle('Where are higher numbers of GCN \n counts expected?', fontsize=25,y=0.91,x=0.45)
plt.xlim([130000, 700000])
plt.ylim([0,670000])
fig.tight_layout()
plt.legend(loc="upper left")  
plt.savefig('./visualisations/GCN_survey_chloropleth_noponds_increases.png', bbox_inches='tight')    
plt.show()
del(ax,fig,base,cmap)

#visualise table
import textwrap as twp
increasestable=merged_county_data[merged_county_data['increases']>0]
increasestable['County'] = increasestable.index
increasestable=increasestable[['County','increases']]
increasestable=increasestable.sort_values('increases',ascending=False)
columns=[twp.fill('County',25), twp.fill('Predicted GCN population increase',25)]
#increasestable.rename(columns={'increases':'Predicted GCN population increase'},inplace=True)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table=ax.table(cellText=increasestable.values, colLabels=columns, loc='center')
table.scale(0.5, 1.5)
plt.rcParams['figure.dpi'] = 300
fig.savefig('./visualisations/increases_table.png', bbox_inches='tight')
plt.show()
