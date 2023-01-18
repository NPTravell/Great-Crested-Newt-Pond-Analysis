<p align="right">Data last refreshed on: 05/04/2022 </p>
<p align="right">Data last analysed on: 16/01/2023</p>
<p align="right">Latest record on: 20/12/2020 </p>

# Great-Crested-Newt-UK-Pond-Analysis

The Great Crested Newt (GCN) is a European protected species and contributes to many ecosystem services and to the health of our environment. The UK has a 'district licensing scheme' that handles newt protection and relocation services on behalf of developers, as previously the relocated GCN populations could end up in well intended but inappropriate "designed ponds"  and inevitably fail due to isolation or poor environmental conditons ([read here](https://freshwaterhabitats.org.uk/projects/newt-conservation/#:~:text=The%20new%20approach%20focuses%20on,newts%20can%20breed%20and%20thrive) or [watch here](https://www.youtube.com/watch?v=efJ0YYD1MbM)). 

This project intends to close the knowledge gap between ambiguous newt population sizes and ideal environmental conditions at a district level via the use of ["surveyed priority pond"](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::surveyed-priority-ponds-england/about) data by Natural England/Defra and analysing Great Crested Newt (GCN) location patterns via the use of machine learning models

The script is intended to be used via the scheduled task manager on a windows remote desktop and the output to function as an automated report (many of the visualisations below are dynamically filtered to top rows in case future datasets have increasing size, for example). The script can be expanded to export datasets to a SQL data warehouse (coming soon) or ported directly to an Altair python dashboard/PowerBI

## When and when are Great Crested Newts found?

Great Crested Newt counts are summed and presented below. Many sightings either did not have dates, or had poor quality (for example, many sightings are in 1905, and it is unclear if that is approriate) so these records will not be counted in the temporal visualisations

![Untitled design](https://user-images.githubusercontent.com/122735369/212549286-e11f6132-33ad-42ec-b2bb-a074f38acf66.jpg)

May, and Wednesdays, are when most GCNs are reported - however this is not indicative of the actual GCN patterns. For instance it may be that GCNS _are_ present more during this time, or it may be that the volunteers gathering the data are more active at this time. 2019 has the highest count of records, though the 2020 data is incomplete (with 17 total records and 0 GCN sightings) - possibly due to the pandemic.

![Untitled design (6)](https://user-images.githubusercontent.com/122735369/212550950-9ac21a6f-07b3-4488-b541-c55c8d491bda.jpg)
<p align="center"><sup>Table showing spatial summary statistics, as well as occupancy proportion. Temporal data is across all sites (no records removed), and spatial data is across all years except when year ('1905') or empty dates were filtered out</sup></p>

Due to the large number individual 'nearest towns' (to the survyed pond) and districts in which ponds were found, the spatial visualisations below are filtered to the top 10 records. Note that district data was extracted seperately via the coordinates of the pond, due to the ambiguous granularity (for example - "Gloucestershire, Wiltshire and Bristol/Bath area" in the original dataset)

![Untitled design (5)](https://user-images.githubusercontent.com/122735369/212550996-275f2d32-39c7-476c-ac56-f4d47f796300.jpg)

The following is a chloropleth map showing which counties have the highest counts of Great Crest Newts (across all years), along showing the locations all pond sites and whether they were found to occupy the site or not

![Untitled design (1)](https://user-images.githubusercontent.com/122735369/212669721-84fe39f2-2ce8-448b-8196-917dca53f4ff.png)
<p align="center"><sup>An interactive map (right) is being developed and will be available soon</sup></p>

No observer variables are available in this dataset, which can bias the dataset and subsequent predictive analytics - for example when a pond is recorded as having no GCNs, it may be that the GCN just wasn't observed, not that it wasn't present, and it is possible that certain observers are more skilled than others. The drastic population size changes seen below may be indicative of this, which highlights a need to standardise data collection methods.

![year_on_year_change](https://user-images.githubusercontent.com/122735369/212964226-8cefec1e-a4d0-4932-8b2e-d636e6a11e67.png)

<p align="center"><sup>This table shows total GCN observations counts in each district, and is ranked by the average of all three years displayed (2017, 2018, and 2019).</sup></p>

## Why are they found there?

To understand drivers of GCN presence, geographical covariates were extracted from publicly available '.tif' files in a 2500m buffer around pond locations, and the median value recorded (along with minimum, maximum, and mean values, though they were not used in the analysis). Data was not available for all covariates across all years so a geographical dataset for the most recent year was used.

Where covariate data was not present in the buffer zone, a district wide average was used, and where a district average was not available, a district wide average was used instead.

Overall, 7 geographical covariates were extracted - [human footprint](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint) (consisting of cumulative human pressure from 8 variables, such as built up environmental and crop land scales), hillshade, elevation, human population, precipitation, windspeed 10m from the ground, and global horizontal irradiance (total solar radiation on a horizontal surface). Data was extracted as a yearly average.

![Untitled design (7)](https://user-images.githubusercontent.com/122735369/212551258-9d945a85-0aea-47fe-a32b-8f498c68ca96.jpg)

<p align="center"><sup>Example visualised geographical tif datasets</sup></p>

Combinations of these variables (for example, elevation alone, or elevation with precipitation), along with a range of machine learning model architecture (such as random forest, XGboost, and logistic regression) were iteratively tested to determine which variables and model would be most accurate.

Each combination was run 5 times and various performance metrics (the boxplot below) were visualised to assist with model selection. Model 732 (using "XGboost" architecture and elevation, global horizontal irradiance, human footprint, hillshade, precipitation, and windspeed data) was chosen due to its accurate performance (approximately 71% accurate) as well as low training and test speeds (not visualised). Model 732 was then fine tuned to acheive increased prediction accuracy (~76% accurate).

![Untitled design (2)](https://user-images.githubusercontent.com/122735369/212969463-2fb9d16e-3df3-42b9-85e2-f220e76166e0.png)

<p align="center"><sup>Top left: model 732 performance predicting GCN presence based on geographical data vs actual GCN presence. Top right: feature importance scores for model 732. Bottom: various performance metrics from the 5 models with the highest accuracy</sup></p>

The most important variables were indices of [human pressure](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint), and the amount of sunlight and rain, whereas the least important variables in this model were hillshade, windspeed and elevation. Future analyses could split apart the human footprint indices to determine the most important factor contained (for example pollution, or navigable waterways), and if possible the data should be collected for each year/month that the pond was survyed in (as human footprint will change as new infrastructure is built, and certain years may have more sunlight).

Climatic variables (except precipitation and sunlight) could potentially be dropped due to both England's temperate climate with few extremes in spatial variations and their lower importance values in the model above.

## Where will they be found in future?

Using the above tuning and variable inputs, model 732 was used to predict occupancy at each site. GCN presence was then summed for each district, and the results displayed below. Note that only positive differences where _more_ GCNs were predicted are highlighed in the right hand visual, as recommending that less attention be paid to ecological surveys would be inappropriate. As a result, we have potential leads on where we can find potentially hidden GCN populations, and we could drive efforts to discover increasing numbers of willing local volunteers.

![Untitled design (3)](https://user-images.githubusercontent.com/122735369/212971941-19e94848-fed2-44cd-b610-88f862843398.png)

## What steps could we take?

This project aimed to better understand the drivers behind Great Crested Newt (GCN) presence at a district level through the use of machine learning and geographical data. The following guidance is provided based on the output of this project:

Conservation steps:
- Ponds in areas with low [human footprint](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint) (an indice of 8 human pressure variables), higher precipitation and less GHI could be prioritised for protection
- Relocation efforts could attempt to find ponds fitting the above criteria
- Volunteer recuitment efforts could be raised in certain districts where GCN presence was predicted to be higher than was recorded
- Data collection could be standardised to avoid drastic changes in GCN population measurement for example:
    - ensuring a level of identification skill
    - striving for more even data collection throughout the year
    - collecting some form of identifier or observer covariates such as years of experience, identification skill, ecological background, visibility on the day of collection, or time spent at pond

Analytical steps:
- Human footprint indices should be split out and each covariate should be analysed to determine what the most important human pressure factors are (for example which has the highest feature importance value - navigable waterways or pollution?)
- Further effort should be made to locate geopgraphical datasets for each year, and each month, to identify more accurate relationships between these covariates and GCN presence
